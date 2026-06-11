"""
inspect_pose.py — standalone pose estimator viewer
Runs multiHMR on a single image or webcam feed and draws the skeleton annotation.

Usage:
    python inspect_pose.py --input path/to/image.jpg
    python inspect_pose.py --input 0                        # webcam index
    python inspect_pose.py --input path/to/video.mp4
    python inspect_pose.py --input image.jpg --save out.png # save result
    python inspect_pose.py --input image.jpg --print-joints # dump raw j2d coords
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from engine.pose_estimation.pose_estimator import PoseEstimator
from inference_videostream5 import Logger

# ══════════════════════════════════════════════════════════════════════════════
# Logger utility
# ══════════════════════════════════════════════════════════════════════════════

class Logger:
    """Lightweight context-manager logger that accumulates named timing splits and notes."""

    def __init__(self, enabled: bool = True):
        self.enabled  = enabled
        self._splits: list[tuple[str, float]] = []
        self._start:  float | None = None
        self._label:  str  = ""
        self._notes: list[str] = []

    def tick(self, label: str) -> "Logger":
        """Call as `with log.tick('label'): ...` or just `log.tick('label')` to start."""
        self._label = label
        return self

    def __enter__(self):
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - self._start
            self._splits.append((self._label, elapsed))

    def stamp(self, label: str, elapsed: float):
        """Manually record a split (for code that already measures its own time)."""
        if self.enabled:
            self._splits.append((label, elapsed))

    def note(self, text: str):
        if self.enabled:
            self._notes.append(text)

    def report(self, path: str | None = None):
        if not self.enabled or not self._splits:
            return
        total = sum(s for _, s in self._splits)
        col   = max(len(l) for l, _ in self._splits) + 2
        sep   = "═" * (col + 26)
        lines = [
            "",
            sep,
            f"  {'RUN LOG':^{col + 22}}",
            sep,
        ]
        for label, secs in self._splits:
            pct    = (secs / total * 100) if total > 0 else 0
            bar_w  = int(pct / 2)
            filled = "█" * bar_w + "░" * (20 - bar_w)
            lines.append(f"  {label:<{col}} {secs:>7.3f}s  [{filled}] {pct:5.1f}%")
        lines += [
            "─" * (col + 26),
            f"  {'TOTAL':<{col}} {total:>7.3f}s",
            "═" * (col + 26),
            "",
        ]
        output = "\n".join(lines)
        print(output)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                if self._notes:
                    f.write("\n".join(self._notes) + "\n\n")   # notes first
                f.write(output + "\n")
            print(f"[info] Log written → {path}")

# ══════════════════════════════════════════════════════════════════════════════
# MJPEG HTTP streamer  (headless-friendly live preview)
# ══════════════════════════════════════════════════════════════════════════════

class MJPEGStreamer:
    """
    Serves rendered frames as an MJPEG stream over HTTP so they can be watched
    in any browser — no display server required.

    Open  http://<host>:<port>/  in your browser to watch.
    """

    def __init__(self, port: int = 8080, jpeg_quality: int = 85):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        self.port    = port
        self.quality = jpeg_quality
        self._frame: bytes = b""          # latest JPEG-encoded frame
        self._lock   = threading.Lock()
        self._stop   = threading.Event()

        streamer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):   # silence access log
                pass

            def do_GET(self):
                if self.path == "/":
                    # Tiny HTML page that auto-loads the stream
                    body = (
                        b"<html><body style='margin:0;background:#000'>"
                        b"<img src='/stream' style='max-width:100%;height:auto'>"
                        b"</body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.end_headers()
                    try:
                        while not streamer._stop.is_set():
                            with streamer._lock:
                                frame = streamer._frame
                            if frame:
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                    + frame + b"\r\n"
                                )
                            time.sleep(0.01)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_response(404)
                    self.end_headers()

        self._server = HTTPServer(("0.0.0.0", port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[stream] Preview → http://localhost:{port}/  (or your server IP)")

    def push(self, rgb_np: np.ndarray):
        """Encode an RGB uint8 frame and make it available to connected clients."""
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self.quality],
        )
        if ok:
            with self._lock:
                self._frame = buf.tobytes()

    def stop(self):
        self._stop.set()
        self._server.shutdown()

# ══════════════════════════════════════════════════════════════════════════════
# Driving-image pose extraction
# ══════════════════════════════════════════════════════════════════════════════

def _build_intrinsic_4x4(focal, princpt) -> torch.Tensor:
    """Build the 4×4 intrinsic matrix used by _load_pose() / animation_infer."""
    K = torch.eye(4, dtype=torch.float32)
    K[0,0] = float(focal[0])
    K[1,1] = float(focal[1])
    K[0,2] = float(princpt[0])
    K[1,2] = float(princpt[1])
    return K


def project2origin_img(target_human, crop_annotation):
    if target_human is None:
        return target_human
    left, top, pad_left, pad_top, scale_factor, crop_size, raw_size = crop_annotation
    device = target_human["loc"].device

    target_human["loc"] = (
        target_human["loc"] - torch.tensor([pad_left, pad_top], device=device)
    ) / scale_factor + torch.tensor([left, top], device=device)

    target_human["dist"] = target_human["dist"] / (crop_size / raw_size)
    return target_human


def _extract_driving_pose(
    driving_path: str,
    pose_estimator: PoseEstimator,
    log: Logger,
    device: str = "cuda",
) -> tuple:
    from engine.pose_estimation.pose_utils.image import img_center_padding
    from engine.pose_estimation.pose_utils.inference_utils import get_camera_parameters
    from engine.pose_estimation.model import forward_model
    from engine.pose_estimation.pose_utils.tracker import bbox_xyxy_to_cxcywh
    from engine.pose_estimation.pose_utils.image import normalize_rgb_tensor
    import torch.nn.functional as F

    PAD_RATIO   = 0.2
    FOV         = 60
    TARGET_SIZE = pose_estimator.mhmr_model.img_size

    # ── 1. Load + pad image ───────────────────────────────────────────────────
    with log.tick("driving | load + pad image"):
        img_bgr = cv2.imread(driving_path)
        img_bgr, offset_w, offset_h = img_center_padding(img_bgr, PAD_RATIO)
        raw_H, raw_W = img_bgr.shape[:2]

    # ── 2. Camera intrinsics ──────────────────────────────────────────────────
    with log.tick("driving | camera intrinsics"):
        raw_K = get_camera_parameters(
            max(raw_H, raw_W), fov=FOV, p_x=None, p_y=None, device=device
        )
        raw_K[..., 0, -1] = raw_W / 2
        raw_K[..., 1, -1] = raw_H / 2

    # ── 3. Crop + resize to TARGET_SIZE ──────────────────────────────────────
    with log.tick("driving | crop + resize for multiHMR"):
        bbox_scaled = bbox_xyxy_to_cxcywh(
            torch.tensor([[0, 0, raw_W, raw_H]], dtype=torch.float32), scale=1.5
        )
        img_tensor_raw = torch.tensor(img_bgr, dtype=torch.float32, device=device)
        img_tensor_raw = img_tensor_raw.unsqueeze(0).permute(0, 3, 1, 2)  # [1,3,H,W]

        cx, cy, bw, bh = bbox_scaled[0]
        left   = max(0, int(cx - bw / 2))
        right  = min(raw_W - 1, int(cx + bw / 2))
        top    = max(0, int(cy - bh / 2))
        bottom = min(raw_H - 1, int(cy + bh / 2))
        crop   = img_tensor_raw[:, :, top:bottom, left:right]

        _, _, h, w   = crop.shape
        scale_factor = min(TARGET_SIZE / w, TARGET_SIZE / h)
        crop         = F.interpolate(crop, scale_factor=scale_factor, mode="bilinear")

        _, _, h, w = crop.shape
        pad_left   = (TARGET_SIZE - w) // 2
        pad_top    = (TARGET_SIZE - h) // 2
        crop       = F.pad(crop, (pad_left, TARGET_SIZE - w - pad_left,
                                   pad_top,  TARGET_SIZE - h - pad_top))
        crop_input = normalize_rgb_tensor(crop)

        crop_annotation = (left, top, pad_left, pad_top, scale_factor,
                           TARGET_SIZE / scale_factor, max(raw_H, raw_W))

        K_model = get_camera_parameters(
            TARGET_SIZE, fov=FOV, p_x=None, p_y=None, device=device
        )

    # ── 4. Run multiHMR ───────────────────────────────────────────────────────
    with log.tick("driving | multiHMR forward pass (pose extraction)"):
        humans = forward_model(
            pose_estimator.mhmr_model, crop_input, K_model,
            pseudo_idx=None, max_dist=None
        )
        if not humans:
            print("[warn] No person detected in driving frame — will reuse last known pose.")
            return None
        human = project2origin_img(humans[0], crop_annotation)

        # ── DEBUG: print once to find the correct joints key ─────────────────────
        import sys
        print(f"[debug] human keys: {list(human.keys())}", file=sys.stderr)
        if 'joints' in human:
            print(f"[debug] human['joints'].shape = {human['joints'].shape}", file=sys.stderr)
        # remove these two prints once confirmed

    # ── 5. Unpack ─────────────────────────────────────────────────────────────
    with log.tick("driving | unpack smplx params + pack tensors"):
        rotvec     = human['rotvec'].float().cpu().detach()              # [53, 3]
        root_pose  = rotvec[0]                                           # [3]
        body_pose  = rotvec[1:22]                                        # [21, 3]
        lhand_pose = rotvec[22:37]                                       # [15, 3]
        rhand_pose = rotvec[37:52]                                       # [15, 3]
        jaw_pose   = rotvec[52]                                          # [3]
        leye_pose  = torch.zeros(3)
        reye_pose  = torch.zeros(3)

        trans = human['transl_pelvis'].float().cpu().detach().squeeze(0) # [3]
        betas = human['shape'].float().cpu().detach()                    # [10]

        expr_raw = human['expression'].float().cpu().detach()            # [10]
        expr     = torch.zeros(100)
        expr[:expr_raw.shape[0]] = expr_raw

        focal_np   = np.array([raw_K[0,0,0].item(), raw_K[0,1,1].item()], dtype=np.float32)
        princpt_np = np.array([raw_W / 2, raw_H / 2],                     dtype=np.float32)
        img_wh_np  = np.array([raw_W, raw_H],                             dtype=np.float32)

        # ── 6. Extract + unproject 2D joints back to original image space ────
        joints_2d_img: np.ndarray | None = None
        if 'j2d' in human:
            j2d = human['j2d'].float().cpu().detach().numpy()  # [J, 2], in crop/model space

            # Invert the crop pipeline from step 3:
            #   crop_annotation = (left, top, pad_left, pad_top, scale_factor, ...)
            left, top, pad_left, pad_top, scale_factor, _, _ = crop_annotation

            j2d_orig = j2d.copy()
            j2d_orig[:, 0] = (j2d[:, 0] - pad_left) / scale_factor + left
            j2d_orig[:, 1] = (j2d[:, 1] - pad_top)  / scale_factor + top

            joints_2d_img = j2d_orig.astype(np.float32)

        def _bf(t):
            return t.unsqueeze(0).unsqueeze(0).to(device)

        smplx_params = {
            "betas":       betas.unsqueeze(0).to(device),
            "root_pose":   _bf(root_pose),
            "body_pose":   _bf(body_pose),
            "jaw_pose":    _bf(jaw_pose),
            "leye_pose":   _bf(leye_pose),
            "reye_pose":   _bf(reye_pose),
            "lhand_pose":  _bf(lhand_pose),
            "rhand_pose":  _bf(rhand_pose),
            "expr":        _bf(expr),
            "trans":       _bf(trans),
            "focal":       _bf(torch.from_numpy(focal_np)),
            "princpt":     _bf(torch.from_numpy(princpt_np)),
            "img_size_wh": _bf(torch.from_numpy(img_wh_np)),
        }

        intr_4x4         = _build_intrinsic_4x4(focal_np, princpt_np)
        render_c2ws      = torch.eye(4).unsqueeze(0).unsqueeze(0).to(device)
        render_intrs     = intr_4x4.unsqueeze(0).unsqueeze(0).to(device)
        render_bg_colors = torch.ones(1, 1, 3, dtype=torch.float32, device=device)

    return smplx_params, render_c2ws, render_intrs, render_bg_colors, joints_2d_img, (raw_W, raw_H)



# ── Skeleton definition (SMPL-X body joints, standard ordering) ───────────────
_SKELETON_EDGES = [
    # spine
    (0, 3),  (3, 6),  (6, 9),  (9, 12), (12, 15),
    # left arm
    (9, 13), (13, 16), (16, 18), (18, 20),
    # right arm
    (9, 14), (14, 17), (17, 19), (19, 21),
    # left leg
    (0, 1),  (1, 4),  (4, 7),  (7, 10),
    # right leg
    (0, 2),  (2, 5),  (5, 8),  (8, 11),
]

_EDGE_COLORS = [
    # spine (5)
    (255, 255,  80), (255, 255,  80), (255, 255,  80), (255, 255,  80), (255, 255,  80),
    # left arm (4)
    ( 80, 200, 255), ( 80, 200, 255), ( 80, 200, 255), ( 80, 200, 255),
    # right arm (4)
    (255, 140,  80), (255, 140,  80), (255, 140,  80), (255, 140,  80),
    # left leg (4)
    (120, 255, 120), (120, 255, 120), (120, 255, 120), (120, 255, 120),
    # right leg (4)
    (220,  80, 255), (220,  80, 255), (220,  80, 255), (220,  80, 255),
]


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_skeleton(
    img_rgb:      np.ndarray,        # [H, W, 3] uint8 RGB
    joints_2d:    np.ndarray,        # [J, 2] float32, in img_rgb pixel space
    joint_radius: int   = 5,
    bone_thick:   int   = 2,
    alpha:        float = 0.85,
) -> np.ndarray:
    overlay = img_rgb.copy()
    num_j   = len(joints_2d)

    for idx, (i, j) in enumerate(_SKELETON_EDGES):
        if i >= num_j or j >= num_j:
            continue
        xi, yi = int(round(joints_2d[i, 0])), int(round(joints_2d[i, 1]))
        xj, yj = int(round(joints_2d[j, 0])), int(round(joints_2d[j, 1]))
        color   = _EDGE_COLORS[idx % len(_EDGE_COLORS)]
        cv2.line(overlay, (xi, yi), (xj, yj), color, bone_thick, cv2.LINE_AA)

    for k in range(num_j):
        x, y = int(round(joints_2d[k, 0])), int(round(joints_2d[k, 1]))
        cv2.circle(overlay, (x, y), joint_radius,     (255, 255, 255), -1,  cv2.LINE_AA)
        cv2.circle(overlay, (x, y), joint_radius - 1, ( 30,  30,  30),  1,  cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, img_rgb, 1.0 - alpha, 0)


def draw_joint_indices(img_rgb: np.ndarray, joints_2d: np.ndarray) -> np.ndarray:
    """Overlay joint index numbers — useful for debugging skeleton edge indices."""
    out = img_rgb.copy()
    for k, (x, y) in enumerate(joints_2d):
        cv2.putText(out, str(k), (int(x) + 4, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 0), 1, cv2.LINE_AA)
    return out


# ── Pose extraction (self-contained, mirrors _extract_driving_pose) ───────────
def extract_pose(frame_bgr, pose_estimator, device):
    from engine.pose_estimation.pose_utils.image import img_center_padding, normalize_rgb_tensor
    from engine.pose_estimation.pose_utils.inference_utils import get_camera_parameters
    from engine.pose_estimation.model import forward_model
    from engine.pose_estimation.pose_utils.tracker import bbox_xyxy_to_cxcywh
    from engine.pose_estimation.pose_utils.image import normalize_rgb_tensor

    PAD_RATIO   = 0.2
    FOV         = 60
    TARGET_SIZE = pose_estimator.mhmr_model.img_size

    # 1. Pad
    img_bgr, offset_w, offset_h = img_center_padding(frame_bgr, PAD_RATIO)
    raw_H, raw_W = img_bgr.shape[:2]

    # 2. Intrinsics
    raw_K = get_camera_parameters(max(raw_H, raw_W), fov=FOV, p_x=None, p_y=None, device=device)
    raw_K[..., 0, -1] = raw_W / 2
    raw_K[..., 1, -1] = raw_H / 2

    # 3. Crop + resize
    bbox_scaled = bbox_xyxy_to_cxcywh(
        torch.tensor([[0, 0, raw_W, raw_H]], dtype=torch.float32), scale=1.5
    )
    img_tensor = torch.tensor(img_bgr, dtype=torch.float32, device=device)
    img_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)  # [1,3,H,W]

    cx, cy, bw, bh = bbox_scaled[0]
    left   = max(0, int(cx - bw / 2))
    right  = min(raw_W - 1, int(cx + bw / 2))
    top    = max(0, int(cy - bh / 2))
    bottom = min(raw_H - 1, int(cy + bh / 2))
    crop   = img_tensor[:, :, top:bottom, left:right]

    _, _, h, w   = crop.shape
    scale_factor = min(TARGET_SIZE / w, TARGET_SIZE / h)
    crop         = F.interpolate(crop, scale_factor=scale_factor, mode="bilinear")

    _, _, h, w = crop.shape
    pad_left   = (TARGET_SIZE - w) // 2
    pad_top    = (TARGET_SIZE - h) // 2
    crop       = F.pad(crop, (pad_left, TARGET_SIZE - w - pad_left,
                               pad_top,  TARGET_SIZE - h - pad_top))
    crop_input = normalize_rgb_tensor(crop)

    crop_annotation = (left, top, pad_left, pad_top, scale_factor,
                       TARGET_SIZE / scale_factor, max(raw_H, raw_W))

    K_model = get_camera_parameters(TARGET_SIZE, fov=FOV, p_x=None, p_y=None, device=device)

    # 4. Run multiHMR
    humans = forward_model(
        pose_estimator.mhmr_model, crop_input, K_model,
        pseudo_idx=None, max_dist=None,
    )
    if not humans:
        return None, (raw_W, raw_H)

    human = project2origin_img(humans[0], crop_annotation)

    # 5. Unproject j2d from crop space → padded image space
    joints_2d = None
    if 'j2d' in human:
        j2d = human['j2d'].float().cpu().detach().numpy()   # [J, 2]
        cl, ct, pl, pt, sf, _, _ = crop_annotation
        j = j2d.copy()
        j[:, 0] = (j[:, 0] - pl) / sf + cl
        j[:, 1] = (j[:, 1] - pt) / sf + ct
        j[:, 0] = np.clip(j[:, 0], 0, raw_W)
        j[:, 1] = np.clip(j[:, 1], 0, raw_H)
        joints_2d = j.astype(np.float32)

    return joints_2d, (raw_W, raw_H)


# ── Render one frame ──────────────────────────────────────────────────────────
def process_frame(frame_bgr, pose_estimator, device, show_indices, print_joints):
    t0 = time.perf_counter()
    joints_2d, (pad_W, pad_H) = extract_pose(frame_bgr, pose_estimator, device)
    elapsed = time.perf_counter() - t0

    # Scale joints from padded-image space → display frame space
    disp_H, disp_W = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if joints_2d is None:
        cv2.putText(frame_rgb, "No person detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 80), 2, cv2.LINE_AA)
        return frame_rgb, elapsed

    # Scale from padded → display
    sx = disp_W / pad_W
    sy = disp_H / pad_H
    j_disp = joints_2d.copy()
    j_disp[:, 0] *= sx
    j_disp[:, 1] *= sy

    if print_joints:
        print("\n[joints_2d] (display-space pixels):")
        for k, (x, y) in enumerate(j_disp):
            print(f"  joint {k:3d}: ({x:7.2f}, {y:7.2f})")

    out = draw_skeleton(frame_rgb, j_disp)

    if show_indices:
        out = draw_joint_indices(out, j_disp)

    # HUD
    cv2.putText(out, f"pose: {elapsed*1000:.0f}ms  joints: {len(j_disp)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"pose: {elapsed*1000:.0f}ms  joints: {len(j_disp)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30),   1, cv2.LINE_AA)

    return out, elapsed


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Standalone multiHMR pose viewer")
    p.add_argument("--input",  required=True,
                   help="Image path, video path, webcam index (0,1,…), or RTSP URL")
    p.add_argument("--model",  default="./pretrained_models/human_model_files",
                   help="Pose estimator model name (default: multiHMR_896_L)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save",   default=None, metavar="PATH",
                   help="Save result image/video to this path")
    p.add_argument("--show-indices", action="store_true",
                   help="Draw joint index numbers on the overlay")
    p.add_argument("--print-joints", action="store_true",
                   help="Print raw j2d coordinates to stdout each frame")
    p.add_argument("--window", action="store_true",
                   help="Show live cv2 window (requires display)")
    p.add_argument("--stream", nargs="?", const="mjpeg", default=None,
                   metavar="MODE", choices=["mjpeg", "window"],
                   help="Stream output: mjpeg = HTTP on :8080 (default), window = cv2.imshow")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load pose estimator ───────────────────────────────────────────────────
    print(f"[info] Loading pose estimator: {args.model}")
    from engine.pose_estimation.pose_estimator import PoseEstimator
    pose_estimator = PoseEstimator(model_dir=args.model, device=args.device)
    print("[info] Model loaded.")

    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    inp         = args.input

    # ── Single image ──────────────────────────────────────────────────────────
    is_image = os.path.isfile(inp) and os.path.splitext(inp)[-1].lower() in _IMAGE_EXTS
    if is_image:
        frame_bgr = cv2.imread(inp)
        out_rgb, elapsed = process_frame(
            frame_bgr, pose_estimator, args.device,
            args.show_indices, args.print_joints,
        )
        print(f"[info] Inference: {elapsed*1000:.1f} ms  |  joints: "
              f"{out_rgb.shape}")

        if args.save:
            from PIL import Image
            Image.fromarray(out_rgb).save(args.save)
            print(f"[✓] Saved → {args.save}")

        if args.window:
            cv2.imshow("inspect_pose", cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif not args.save:
            # No window, no save — at least write a temp file so user can see it
            tmp = "/tmp/inspect_pose_out.png"
            from PIL import Image
            Image.fromarray(out_rgb).save(tmp)
            print(f"[info] No --save or --window given. Written to {tmp}")
        return

    # ── Video / webcam ────────────────────────────────────────────────────────
    try:
        src = int(inp)          # webcam index
    except ValueError:
        src = inp               # file path or RTSP URL

    if isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(src)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)

    _streamer = None
    if args.stream == "mjpeg":
        _streamer = MJPEGStreamer()
        print("[stream] MJPEG streamer started — open http://localhost:8080 in your browser")
    elif args.stream == "window":
        print("[stream] cv2.imshow window — press Q or Esc to quit.")
        
    frame_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            out_rgb, elapsed = process_frame(
                frame_bgr, pose_estimator, args.device,
                args.show_indices, args.print_joints,
            )

            label = f"Frame {frame_idx+1}" + (f"/{total}" if total > 0 else "")
            print(f"[info] {label}  pose={elapsed*1000:.0f}ms")

            # Lazy writer init
            if writer is None and args.save:
                oh, ow = out_rgb.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save, fourcc, fps, (ow, oh))

            if writer is not None:
                writer.write(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

            if args.stream == "mjpeg" and _streamer is not None:
                _streamer.push(out_rgb)
            elif args.stream == "window" or args.window:
                cv2.imshow("inspect_pose", cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("[info] Quit.")
                    break

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[info] Interrupted.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            if args.save and os.path.exists(args.save):
                import shutil, subprocess
                if shutil.which("ffmpeg"):
                    tmp_raw = args.save.replace(".mp4", "_raw.mp4")
                    os.rename(args.save, tmp_raw)
                    r = subprocess.run([
                        "ffmpeg", "-y", "-i", tmp_raw,
                        "-vcodec", "libx264", "-preset", "fast",
                        "-crf", "23", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", args.save,
                    ], capture_output=True)
                    if r.returncode == 0:
                        os.remove(tmp_raw)
                        print(f"[✓] Re-encoded → {args.save}")
                    else:
                        os.rename(tmp_raw, args.save)
        if _streamer is not None:
            _streamer.stop()
        if args.stream == "window" or args.window:
            cv2.destroyAllWindows()

    print(f"[✓] Done. {frame_idx} frames processed.")


if __name__ == "__main__":
    main()