#!/usr/bin/env python3
"""
inference.py  –  LHM single-image retargeting pipeline

Given a SOURCE portrait and a DRIVING portrait:
  1. Segment + preprocess the source image.
  2. Build a 3-D Gaussian avatar from the source image via LHM.
  3. Extract SMPL-X pose from the driving image via multiHMR.
  4. Render the source avatar in the driving pose.
  5. Composite the result (white bg, or onto the driving image background).

Usage
-----
python inference.py \
    --source  ./train_data/example_imgs/00000000_joker_2.jpg \
    --driving ./train_data/example_imgs/mimo.jpg \
    --output  ./outputs/ \
    [--model_name LHM-1B] \
    [--bg white|driving] \
    [--device cuda] \
    [--timer]
    [--stream [mjpeg|window|rtsp]]
    [--save-avatar true|false]

eg: python inference_videostream2.py --source ./train_data/example_imgs/00000000_joker_2.jpg --driving ./train_data/example_imgs/mimo2.mp4 --output --timer
"""

import argparse
import os
import time
from collections import defaultdict

# Must be initialized before any LHM/accelerate imports,
# otherwise accelerate.logging raises "You must initialize the accelerate state".
from accelerate import PartialState
PartialState()

import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

# ── segmentation ─────────────────────────────────────────────────────────────
try:
    from engine.SegmentAPI.SAM import SAM2Seg
    _HAS_SAM2 = True
except Exception:
    print("\033[33m[warn] SAM2 not found – falling back to rembg.\033[0m")
    from rembg import remove as rembg_remove
    _HAS_SAM2 = False

# ── pose / face ───────────────────────────────────────────────────────────────
from engine.pose_estimation.pose_estimator import PoseEstimator
from engine.SegmentAPI.base import Bbox

# ── LHM utilities ─────────────────────────────────────────────────────────────
from LHM.utils.model_download_utils import AutoModelQuery
from LHM.runners.infer.utils import (
    calc_new_tgt_size_by_aspect,
    center_crop_according_to_mask,
    resize_image_keepaspect_np,
)
from LHM.utils.download_utils import download_extract_tar_from_url, download_from_url
from LHM.utils.face_detector import FaceDetector          # used in human_lrm.py
from LHM.utils.hf_hub import wrap_model_hub
from LHM.utils.model_card import MEMORY_MODEL_CARD, MODEL_CARD, MODEL_CONFIG
from LHM.utils.model_query_utils import AutoModelSwitcher

# ─────────────────────────────────────────────────────────────────────────────
ASPECT_STANDARD: float = 5.0 / 3.0


# ══════════════════════════════════════════════════════════════════════════════
# Avatar cache  (save / load 3-D Gaussian avatar to disk)
# ══════════════════════════════════════════════════════════════════════════════

_AVATAR_FILES = ("gs_model_list.pt", "query_points.pt", "transform_mat.pt", "src_betas.pt")


def _avatar_cache_dir(source_path: str) -> str:
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join("./avatars", stem)


def _avatar_cache_exists(source_path: str) -> bool:
    d = _avatar_cache_dir(source_path)
    return all(os.path.exists(os.path.join(d, f)) for f in _AVATAR_FILES)


def _save_avatar(source_path: str, gs_model_list, query_points, transform_mat, src_betas):
    d = _avatar_cache_dir(source_path)
    os.makedirs(d, exist_ok=True)
    torch.save(gs_model_list, os.path.join(d, "gs_model_list.pt"))
    torch.save(query_points,  os.path.join(d, "query_points.pt"))
    torch.save(transform_mat, os.path.join(d, "transform_mat.pt"))
    torch.save(src_betas,     os.path.join(d, "src_betas.pt"))
    print(f"[✓] Avatar cached → {d}/")


def _load_avatar(source_path: str, device: str):
    d = _avatar_cache_dir(source_path)
    gs_model_list = torch.load(os.path.join(d, "gs_model_list.pt"), map_location=device)
    query_points  = torch.load(os.path.join(d, "query_points.pt"),  map_location=device)
    transform_mat = torch.load(os.path.join(d, "transform_mat.pt"), map_location=device)
    src_betas     = torch.load(os.path.join(d, "src_betas.pt"),     map_location=device)
    print(f"[✓] Avatar loaded from cache → {d}/")
    return gs_model_list, query_points, transform_mat, src_betas


# ══════════════════════════════════════════════════════════════════════════════
# Timer utility
# ══════════════════════════════════════════════════════════════════════════════

class Timer:
    """Lightweight context-manager timer that accumulates named splits."""

    def __init__(self, enabled: bool = True):
        self.enabled  = enabled
        self._splits: list[tuple[str, float]] = []
        self._start:  float | None = None
        self._label:  str  = ""

    def tick(self, label: str) -> "Timer":
        """Call as `with timer.tick('label'): ...` or just `timer.tick('label')` to start."""
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

    def report(self, path: str | None = None):
        if not self.enabled or not self._splits:
            return
        total = sum(s for _, s in self._splits)
        col   = max(len(l) for l, _ in self._splits) + 2
        sep   = "═" * (col + 26)
        lines = [
            "",
            sep,
            f"  {'TIMING REPORT':^{col + 22}}",
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
                f.write(output + "\n")
            print(f"[info] Timings written → {path}")


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
# RTSP streamer  (ffmpeg subprocess)
# ══════════════════════════════════════════════════════════════════════════════

class RTSPStreamer:
    """
    Pushes rendered frames into a local RTSP server via ffmpeg.

    Requires ffmpeg with libx264.  Start a local RTSP server first, e.g.:
        docker run --rm -it -p 8554:8554 aler9/rtsp-simple-server
    Then connect any RTSP player (VLC, ffplay) to:
        rtsp://localhost:8554/live

    If no external server is available, falls back to serving the raw H.264
    stream over TCP so ffplay can consume it directly:
        ffplay tcp://localhost:8554?listen
    """

    def __init__(self, port: int = 8554, fps: float = 25.0):
        import subprocess, shutil
        self.fps  = fps
        self.port = port
        self._proc: subprocess.Popen | None = None
        self._size: tuple[int, int] | None  = None   # (w, h) set on first push
        self._ffmpeg = shutil.which("ffmpeg")
        if self._ffmpeg is None:
            raise RuntimeError("ffmpeg not found in PATH — required for RTSP streaming.")

    def _start(self, w: int, h: int):
        import subprocess
        cmd = [
            self._ffmpeg, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-vcodec", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-f", "rtsp", f"rtsp://localhost:{self.port}/live",
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._size = (w, h)
        print(f"[stream] RTSP → rtsp://localhost:{self.port}/live")
        print(f"[stream] Watch with:  ffplay rtsp://localhost:{self.port}/live")
        print(f"[stream]          or: vlc rtsp://localhost:{self.port}/live")

    def push(self, rgb_np: np.ndarray):
        h, w = rgb_np.shape[:2]
        if self._proc is None:
            self._start(w, h)
        bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        try:
            self._proc.stdin.write(bgr.tobytes())
        except BrokenPipeError:
            pass

    def stop(self):
        if self._proc and self._proc.stdin:
            self._proc.stdin.close()
            self._proc.wait()


# ══════════════════════════════════════════════════════════════════════════════
# Setup helpers
# ══════════════════════════════════════════════════════════════════════════════

def _download_geo_files() -> None:
    dst = "./pretrained_models/dense_sample_points/1_20000.ply"
    if not os.path.exists(dst):
        download_from_url(
            "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com"
            "/share/aigc3d/data/LHM/1_20000.ply",
            "./pretrained_models/dense_sample_points/",
        )


def _prior_check() -> None:
    if not os.path.exists("./pretrained_models"):
        download_extract_tar_from_url(MODEL_CARD["prior_model"])


def _query_model_config(model_name: str):
    try:
        params = model_name.split("-")[1]
        return MODEL_CONFIG[params]
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Image preprocessing  (mirrors infer_preprocess_image in human_lrm.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

def _get_bbox(mask: np.ndarray) -> "Bbox":
    h, w = mask.shape
    pha = (mask / 255.0)
    pha[pha < 0.5]  = 0.0
    pha[pha >= 0.5] = 1.0
    _h, _w = np.where(pha == 1)
    whwh = [_w.min().item(), _h.min().item(), _w.max().item(), _h.max().item()]
    return Bbox(whwh).scale(1.1, width=w, height=h)


def _infer_preprocess_image(
    rgb_path: str,
    mask: np.ndarray,
    *,
    max_tgt_size: int,
    aspect_standard: float,
    render_tgt_size: int,
    bg_color: float = 1.0,
    multiply: int = 14,
):
    """Returns rgb [1,3,H,W] and mask [1,1,H,W] tensors, float32, values in [0,1]."""
    rgb = np.array(Image.open(rgb_path))

    bbox      = _get_bbox(mask)
    x0, y0, x1, y1 = bbox.get_box()
    rgb  = rgb [y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]

    h, w, _ = rgb.shape
    assert w < h, (
        f"Cropped region is wider ({w}) than tall ({h}). "
        "Ensure the image shows a full upright body."
    )

    cur_ratio   = h / w
    scale_ratio = cur_ratio / aspect_standard
    target_w    = int(min(w * scale_ratio, h))

    if target_w > w:
        off = (target_w - w) // 2
        rgb  = np.pad(rgb,  ((0,0),(off,off),(0,0)), constant_values=255)
        mask = np.pad(mask, ((0,0),(off,off)),        constant_values=0)
    else:
        target_h = int(w * aspect_standard)
        off_h    = target_h - h
        rgb  = np.pad(rgb,  ((off_h,0),(0,0),(0,0)), constant_values=255)
        mask = np.pad(mask, ((off_h,0),(0,0)),        constant_values=0)

    rgb  = rgb / 255.0
    mask = (mask / 255.0 > 0.5).astype(np.float32)
    rgb  = rgb[:,:,:3] * mask[:,:,None] + bg_color * (1 - mask[:,:,None])

    rgb  = resize_image_keepaspect_np(rgb,  max_tgt_size)
    mask = resize_image_keepaspect_np(mask, max_tgt_size)

    rgb, mask, _, _ = center_crop_according_to_mask(
        rgb, mask, aspect_standard, [1.0, 1.0]
    )

    tgt_hw, _, _ = calc_new_tgt_size_by_aspect(
        cur_hw=rgb.shape[:2],
        aspect_standard=aspect_standard,
        tgt_size=render_tgt_size,
        multiply=multiply,
    )
    rgb  = cv2.resize(rgb,  (tgt_hw[1], tgt_hw[0]), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (tgt_hw[1], tgt_hw[0]), interpolation=cv2.INTER_AREA)

    rgb_t  = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0)
    mask_t = torch.from_numpy(mask[:,:,None]).float().permute(2,0,1).unsqueeze(0)
    return rgb_t, mask_t


# ══════════════════════════════════════════════════════════════════════════════
# Segmentation
# ══════════════════════════════════════════════════════════════════════════════

def _segment(image_path: str, parsing_net) -> np.ndarray:
    """Return uint8 H×W mask (0/255) for the foreground person."""
    if parsing_net is not None:
        out = parsing_net(img_path=image_path, bbox=None)
        return (out.masks * 255).astype(np.uint8)
    else:
        img_np  = cv2.imread(image_path)
        removed = rembg_remove(img_np)
        return removed[..., 3]


# ══════════════════════════════════════════════════════════════════════════════
# Face crop  (mirrors crop_face_image in human_lrm.py)
# ══════════════════════════════════════════════════════════════════════════════

def _crop_face(rgb_path: str, face_detector, src_head_size: int) -> torch.Tensor:
    """Returns [1,3,H,W] float32 tensor in [0,1]."""
    rgb   = np.array(Image.open(rgb_path))[..., :3]
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)   # [3,H,W]
    try:
        bbox     = face_detector(rgb_t)               # FaceDetector is callable
        head_rgb = rgb_t[:, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        head_np  = head_rgb.permute(1, 2, 0).numpy()
    except Exception:
        print("[warn] face detection failed – using blank head crop.")
        head_np = np.zeros((src_head_size, src_head_size, 3), dtype=np.uint8)

    try:
        head_np = cv2.resize(
            head_np, (src_head_size, src_head_size), interpolation=cv2.INTER_AREA
        )
    except Exception:
        head_np = np.zeros((src_head_size, src_head_size, 3), dtype=np.uint8)

    return torch.from_numpy(head_np / 255.0).float().permute(2,0,1).unsqueeze(0)


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
    timer: Timer,
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
    with timer.tick("driving | load + pad image"):
        img_bgr = cv2.imread(driving_path)
        img_bgr, offset_w, offset_h = img_center_padding(img_bgr, PAD_RATIO)
        raw_H, raw_W = img_bgr.shape[:2]

    # ── 2. Camera intrinsics ──────────────────────────────────────────────────
    with timer.tick("driving | camera intrinsics"):
        raw_K = get_camera_parameters(
            max(raw_H, raw_W), fov=FOV, p_x=None, p_y=None, device=device
        )
        raw_K[..., 0, -1] = raw_W / 2
        raw_K[..., 1, -1] = raw_H / 2

    # ── 3. Crop + resize to TARGET_SIZE ──────────────────────────────────────
    with timer.tick("driving | crop + resize for multiHMR"):
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
    with timer.tick("driving | multiHMR forward pass (pose extraction)"):
        humans = forward_model(
            pose_estimator.mhmr_model, crop_input, K_model,
            pseudo_idx=None, max_dist=None
        )
        if not humans:
            print("[warn] No person detected in driving frame — will reuse last known pose.")
            return None
        human = project2origin_img(humans[0], crop_annotation)

    # ── 5. Unpack ─────────────────────────────────────────────────────────────
    with timer.tick("driving | unpack smplx params + pack tensors"):
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

    return smplx_params, render_c2ws, render_intrs, render_bg_colors


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def _parse_configs(model_name: str):
    query_model        = AutoModelQuery()
    model_path         = query_model.query(model_name)
    cli_cfg            = OmegaConf.create()
    cfg                = OmegaConf.create()
    cli_cfg.model_name = model_path

    model_config = _query_model_config(model_name)
    cfg_train    = None
    if model_config is not None:
        cfg_train         = OmegaConf.load(model_config)
        cfg.source_size   = cfg_train.dataset.source_image_res
        cfg.render_size   = cfg_train.dataset.render_image.high
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except Exception:
            cfg.src_head_size = 112
    else:
        cfg.source_size   = 512
        cfg.render_size   = 512
        cfg.src_head_size = 112

    cfg.merge_with(cli_cfg)
    return cfg, cfg_train


def _build_model(cfg):
    from LHM.models import model_dict
    hf_cls = wrap_model_hub(model_dict["human_lrm_sapdino_bh_sd3_5"])
    return hf_cls.from_pretrained(cfg.model_name)


# ══════════════════════════════════════════════════════════════════════════════
# Avatar builder  (source image only — runs once before any driving frames)
# ══════════════════════════════════════════════════════════════════════════════

def _build_avatar(
    source_path:   str,
    lhm,
    pose_estimator,
    face_detector,
    parsing_net,
    source_size:   int,
    src_head_size: int,
    dtype:         torch.dtype,
    device:        str,
    timer:         Timer,
    # A single driving frame's camera params are needed by infer_single_view.
    # We pass a neutral/identity placeholder here; the actual per-frame camera
    # is used during animation_infer, not during avatar construction.
    render_c2ws:      torch.Tensor,
    render_intrs:     torch.Tensor,
    render_bg_colors: torch.Tensor,
    smplx_params_ref: dict,
) -> tuple:
    """
    Build the 3-D Gaussian avatar from the source image.
    Returns (gs_model_list, query_points, transform_mat_neutral_pose, src_betas).
    src_betas is the body-shape tensor to override driving-frame betas with.
    """
    # ── Source segmentation ───────────────────────────────────────────────────
    print("[info] Preprocessing source image …")
    with timer.tick("source  | segmentation (SAM2 / rembg mask)"):
        src_mask = _segment(source_path, parsing_net)

    with timer.tick("source  | preprocess + bg removal + resize"):
        src_rgb, _ = _infer_preprocess_image(
            source_path, src_mask,
            max_tgt_size=896,
            aspect_standard=ASPECT_STANDARD,
            render_tgt_size=source_size,
            bg_color=1.0,
            multiply=14,
        )

    with timer.tick("source  | face crop (head region)"):
        src_head = _crop_face(source_path, face_detector, src_head_size)

    with timer.tick("source  | body shape estimation (beta override)"):
        src_shape = pose_estimator(source_path)
        if src_shape.is_full_body and src_shape.beta is not None:
            src_betas = torch.tensor(src_shape.beta, dtype=dtype).unsqueeze(0).to(device)
            print("[info] Using source person's body shape (beta).")
        else:
            src_betas = None
            print("[warn] Could not estimate source body shape – will use driving betas.")

    # ── Build avatar ──────────────────────────────────────────────────────────
    print("[info] Building 3-D avatar from source …")
    lhm.to(dtype)
    with timer.tick("avatar  | build 3-D Gaussians from source (infer_single_view)"):
        with torch.no_grad():
            gs_model_list, query_points, transform_mat_neutral_pose = lhm.infer_single_view(
                src_rgb.unsqueeze(0).to(device, dtype),
                src_head.unsqueeze(0).to(device, dtype),
                None, None,
                render_c2ws=render_c2ws,
                render_intrs=render_intrs,
                render_bg_colors=render_bg_colors,
                smplx_params={k: v.to(device) for k, v in smplx_params_ref.items()},
            )

    return gs_model_list, query_points, transform_mat_neutral_pose, src_betas


# ══════════════════════════════════════════════════════════════════════════════
# Per-frame renderer
# ══════════════════════════════════════════════════════════════════════════════

def _render_frame(
    frame_bgr:                  np.ndarray,
    frame_path_tmp:             str,
    lhm,
    pose_estimator,
    gs_model_list,
    query_points,
    transform_mat_neutral_pose,
    src_betas:                  torch.Tensor | None,
    dtype:                      torch.dtype,
    device:                     str,
    bg_mode:                    str,
    timer:                      Timer,
) -> np.ndarray:
    """
    Given a single driving frame (as a BGR numpy array), extract pose and
    render the avatar.  Returns the composited output as an RGB numpy array.
    frame_path_tmp is a temp file path used by _extract_driving_pose (which
    expects a file path).
    """
    # Write frame to temp file so _extract_driving_pose can read it
    cv2.imwrite(frame_path_tmp, frame_bgr)

    # ── Pose extraction ───────────────────────────────────────────────────────
    t_pose_start = time.perf_counter()
    pose_result = _extract_driving_pose(frame_path_tmp, pose_estimator, timer, device=device)
    timer.stamp("driving | TOTAL (pose extraction)", time.perf_counter() - t_pose_start)

    if pose_result is None:
        # No person detected — return the original driving frame unchanged
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    smplx_params, render_c2ws, render_intrs, render_bg_colors = pose_result

    # Override betas with source body shape
    if src_betas is not None:
        smplx_params["betas"] = src_betas

    smplx_params["transform_mat_neutral_pose"] = transform_mat_neutral_pose

    # ── Render ────────────────────────────────────────────────────────────────
    with timer.tick("render  | apply driving pose to avatar (animation_infer)"):
        with torch.no_grad():
            res = lhm.animation_infer(
                gs_model_list, query_points, smplx_params,
                render_c2ws=render_c2ws,
                render_intrs=render_intrs,
                render_bg_colors=render_bg_colors,
            )

    # ── Composite ─────────────────────────────────────────────────────────────
    with timer.tick("output  | composite frame"):
        comp_rgb  = res["comp_rgb"]
        comp_mask = res["comp_mask"]
        comp_mask[comp_mask < 0.5] = 0.0

        rendered_np = (
            (comp_rgb * comp_mask + (1 - comp_mask) * 1.0)
            .clamp(0, 1)[0].cpu().numpy() * 255
        ).astype(np.uint8)

        if bg_mode == "driving":
            drv_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            dh, dw   = drv_rgb.shape[:2]
            rend_res = cv2.resize(rendered_np, (dw, dh), interpolation=cv2.INTER_LANCZOS4)
            mask_np  = comp_mask[0].clamp(0, 1).cpu().numpy()
            if mask_np.ndim == 3 and mask_np.shape[-1] != 1:
                mask_np = mask_np[..., :1]
            mask_res = cv2.resize(
                mask_np.squeeze(-1) if mask_np.ndim == 3 else mask_np,
                (dw, dh), interpolation=cv2.INTER_LANCZOS4,
            )[:, :, None]
            output_img = (rend_res * mask_res + drv_rgb * (1 - mask_res)).clip(0, 255).astype(np.uint8)
        else:
            output_img = rendered_np

    return output_img


def _vram_mb() -> float:
    """Current GPU memory allocated in MB, or 0 if no CUDA."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def _model_label(tag: str, name: str, vram_before: float, vram_after: float) -> str:
    """Build a timer label that includes model name + VRAM delta."""
    delta = vram_after - vram_before
    vram_str = f"+{delta:.0f} MB VRAM" if delta > 0 else "CPU only"
    return f"model load | {tag} [{name}]  ({vram_str})"


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(
    source_path:   str,
    driving:       str,           # image path, video path, or stream URL / index
    output_path:   str,           # image path for single frame; directory for video
    model_name:    str  = "LHM-1B",
    bg_mode:       str  = "white",
    device:        str  = "cuda",
    timer_path:    str | None = None,   # None=disabled, path=write timings there
    stream_mode:   str | None = None,   # None | 'mjpeg' | 'window' | 'rtsp'
    save_avatar:   bool = True,         # cache built avatar to ./avatars/{source}/
) -> str:
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    timer = Timer(enabled=timer_path is not None)
    t_total_start = time.perf_counter()

    # ── 0. Downloads ──────────────────────────────────────────────────────────
    with timer.tick("downloads / prior check"):
        _download_geo_files()
        _prior_check()

    # ── 1. Auto-select model ──────────────────────────────────────────────────
    switcher   = AutoModelSwitcher(MEMORY_MODEL_CARD, extra_memory=0)
    model_name = switcher.query(model_name)
    print(f"[info] Model: {model_name}")

    os.environ.update({
        "APP_ENABLED":           "1",
        "APP_MODEL_NAME":         model_name,
        "APP_TYPE":              "infer.human_lrm",
        "NUMBA_THREADING_LAYER": "omp",
    })

    # ── 2. Check avatar cache before deciding which models to load ────────────
    use_cached_avatar = _avatar_cache_exists(source_path)
    if use_cached_avatar:
        print(
            f"[info] Cached avatar found for '{os.path.basename(source_path)}' "
            "— skipping face detector + segmentation."
        )

    # ── 3. Load sub-models ────────────────────────────────────────────────────
    print("[info] Loading pose estimator …")
    _vram0 = _vram_mb()
    with timer.tick("_pose_estimator_placeholder_"):
        pose_estimator = PoseEstimator(
            "./pretrained_models/human_model_files/", device="cpu"
        )
        pose_estimator.to(device)
        pose_estimator.device = device
    _vram1 = _vram_mb()
    try:
        _pe_name = type(pose_estimator.mhmr_model).__name__
    except Exception:
        _pe_name = "PoseEstimator"
    timer._splits[-1] = (_model_label("pose estimator", _pe_name, _vram0, _vram1), timer._splits[-1][1])

    # Face detector and segmentation are only needed to build the avatar.
    # Skip them entirely when a cached avatar is available.
    if not use_cached_avatar:
        print("[info] Loading face detector …")
        _vram0 = _vram_mb()
        with timer.tick("_face_detector_placeholder_"):
            face_detector = FaceDetector(
                "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
                device=device,
            )
        _vram1 = _vram_mb()
        try:
            _fd_name = os.path.basename(
                "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd"
            )
        except Exception:
            _fd_name = "FaceDetector"
        timer._splits[-1] = (_model_label("face detector", _fd_name, _vram0, _vram1), timer._splits[-1][1])

        print("[info] Loading segmentation …")
        _vram0 = _vram_mb()
        with timer.tick("_seg_placeholder_"):
            parsing_net = SAM2Seg() if _HAS_SAM2 else None
        _vram1 = _vram_mb()
        _seg_name = "SAM2" if _HAS_SAM2 else "rembg"
        timer._splits[-1] = (_model_label("segmentation", _seg_name, _vram0, _vram1), timer._splits[-1][1])
    else:
        face_detector = None
        parsing_net   = None

    cfg, _ = _parse_configs(model_name)

    print("[info] Loading LHM …")
    _vram0 = _vram_mb()
    with timer.tick("_lhm_placeholder_"):
        lhm = _build_model(cfg)
        lhm.to(device)
        lhm.eval()
    _vram1 = _vram_mb()
    try:
        _lhm_name = f"{type(lhm).__name__} ({model_name})"
    except Exception:
        _lhm_name = model_name
    timer._splits[-1] = (_model_label("LHM", _lhm_name, _vram0, _vram1), timer._splits[-1][1])

    source_size   = cfg.source_size
    src_head_size = cfg.src_head_size
    dtype         = torch.float32

    # ── 4. Decide driving mode: single image vs video/stream ─────────────────
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    driving_ext = os.path.splitext(driving)[-1].lower()
    is_image    = driving_ext in _IMAGE_EXTS and os.path.isfile(driving)

    # For a stream/int index (e.g. "0"), convert to int so cv2 accepts it
    if not is_image:
        try:
            driving_cv = int(driving)   # webcam index
        except ValueError:
            driving_cv = driving        # file path or RTSP URL

    # ── 5. Neutral camera placeholders for infer_single_view ─────────────────
    neutral_c2w  = torch.eye(4).unsqueeze(0).unsqueeze(0).to(device)
    neutral_intr = _build_intrinsic_4x4([500., 500.], [256., 256.])
    neutral_intr = neutral_intr.unsqueeze(0).unsqueeze(0).to(device)
    neutral_bg   = torch.ones(1, 1, 3, dtype=torch.float32, device=device)

    # Minimal placeholder smplx_params (neutral T-pose) for infer_single_view
    def _zeros_bf(*shape):
        return torch.zeros(*shape, dtype=dtype).unsqueeze(0).unsqueeze(0).to(device)

    neutral_smplx = {
        "betas":       torch.zeros(1, 10, dtype=dtype, device=device),
        "root_pose":   _zeros_bf(3),
        "body_pose":   _zeros_bf(21, 3),
        "jaw_pose":    _zeros_bf(3),
        "leye_pose":   _zeros_bf(3),
        "reye_pose":   _zeros_bf(3),
        "lhand_pose":  _zeros_bf(15, 3),
        "rhand_pose":  _zeros_bf(15, 3),
        "expr":        _zeros_bf(100),
        "trans":       _zeros_bf(3),
        "focal":       _zeros_bf(2),
        "princpt":     _zeros_bf(2),
        "img_size_wh": _zeros_bf(2),
    }

    # ── 6. Build or load avatar (source only — once) ─────────────────────────
    if use_cached_avatar:
        with timer.tick("avatar  | load from cache"):
            gs_model_list, query_points, transform_mat_neutral_pose, src_betas = \
                _load_avatar(source_path, device)
    else:
        gs_model_list, query_points, transform_mat_neutral_pose, src_betas = _build_avatar(
            source_path    = source_path,
            lhm            = lhm,
            pose_estimator = pose_estimator,
            face_detector  = face_detector,
            parsing_net    = parsing_net,
            source_size    = source_size,
            src_head_size  = src_head_size,
            dtype          = dtype,
            device         = device,
            timer          = timer,
            render_c2ws      = neutral_c2w,
            render_intrs     = neutral_intr,
            render_bg_colors = neutral_bg,
            smplx_params_ref = neutral_smplx,
        )
        if save_avatar:
            with timer.tick("avatar  | save to cache"):
                _save_avatar(
                    source_path, gs_model_list, query_points,
                    transform_mat_neutral_pose, src_betas,
                )

    # ── 7. Single-image mode ──────────────────────────────────────────────────
    if is_image:
        frame_bgr  = cv2.imread(driving)
        tmp_path   = driving          # already a file — reuse directly
        output_img = _render_frame(
            frame_bgr, tmp_path, lhm, pose_estimator,
            gs_model_list, query_points, transform_mat_neutral_pose,
            src_betas, dtype, device, bg_mode, timer,
        )
        if output_path is not None:
            out_path = output_path if not (output_path.endswith("/") or os.path.isdir(output_path)) \
                       else os.path.join(output_path, "result.png")
            Image.fromarray(output_img).save(out_path)
            print(f"[✓] Saved → {out_path}")
        else:
            print("[info] Output not saved (no --output given).")

    # ── 8. Video / stream mode ────────────────────────────────────────────────
    else:
        import tempfile

        cap = cv2.VideoCapture(driving_cv)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open driving source: {driving!r}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # -1 for streams
        is_stream = (total <= 0)

        # Output: directory → write numbered PNGs + one MP4
        if output_path is None:
            out_dir   = None
            out_video = None
        elif output_path.endswith("/") or os.path.isdir(output_path):
            out_dir   = output_path
            out_video = os.path.join(out_dir, "output.mp4")
        else:
            out_dir   = os.path.dirname(output_path) or "."
            out_video = output_path if output_path.endswith(".mp4") \
                        else os.path.splitext(output_path)[0] + ".mp4"
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)

        # Lazy VideoWriter init — size known only after first rendered frame.
        # Try H.264 encoders in order (best Windows/macOS compatibility), fall
        # back to MPEG-4 if none are available in this OpenCV build.
        def _make_writer(path: str, fps: float, size: tuple) -> cv2.VideoWriter:
            for fourcc_str in ("avc1", "H264", "X264", "mp4v"):
                w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_str), fps, size)
                if w.isOpened():
                    print(f"[info] Video encoder: {fourcc_str}")
                    return w
                w.release()
            raise RuntimeError("No working video encoder found (tried avc1/H264/X264/mp4v).")

        writer: cv2.VideoWriter | None = None

        # Initialise the requested streamer (or none)
        _streamer = None
        if stream_mode == "mjpeg":
            _streamer = MJPEGStreamer()
        elif stream_mode == "rtsp":
            _streamer = RTSPStreamer(fps=fps)
        elif stream_mode == "window":
            print("[stream] cv2.imshow window — press Q or Esc to quit.")

        frame_idx  = 0
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(tmp_fd)

        print(f"[info] Starting video loop {'(stream)' if is_stream else f'({total} frames)'} …")
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                print(f"[info] Frame {frame_idx + 1}" +
                      (f"/{total}" if not is_stream else "") + " …")

                with timer.tick(f"frame {frame_idx:05d} | total"):
                    output_img = _render_frame(
                        frame_bgr, tmp_path, lhm, pose_estimator,
                        gs_model_list, query_points, transform_mat_neutral_pose,
                        src_betas, dtype, device, bg_mode, timer,
                    )

                # Lazy VideoWriter init (size known after first render)
                if writer is None and out_video is not None:
                    oh, ow = output_img.shape[:2]
                    writer = _make_writer(out_video, fps, (ow, oh))

                # Write frame to video (VideoWriter expects BGR)
                if writer is not None:
                    with timer.tick(f"frame {frame_idx:05d} | write to video"):
                        writer.write(cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

                # Live preview
                if stream_mode == "mjpeg" and _streamer is not None:
                    _streamer.push(output_img)
                elif stream_mode == "rtsp" and _streamer is not None:
                    _streamer.push(output_img)
                elif stream_mode == "window":
                    cv2.imshow("LHM Output", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("\n[info] Window closed by user.")
                        break

                frame_idx += 1

        except KeyboardInterrupt:
            print("\n[info] Interrupted — saving output so far …")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if _streamer is not None:
                _streamer.stop()
            if stream_mode == "window":
                cv2.destroyAllWindows()
            os.unlink(tmp_path)

        if out_video is not None:
            print(f"[✓] Video saved → {out_video}  ({frame_idx} frames)")
        else:
            print(f"[info] {frame_idx} frames processed — output not saved (no --output given).")

    # ── 9. Timing report ──────────────────────────────────────────────────────
    timer.stamp("── WALL CLOCK TOTAL ──", time.perf_counter() - t_total_start)
    if timer_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(timer_path)) or ".", exist_ok=True)
        timer.report(path=timer_path)

    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="LHM retargeting: transfer source identity to driving pose.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--source",  required=True,
                   help="Source image path (identity donor).")
    p.add_argument("--driving", required=True,
                   help=(
                       "Driving input — one of:\n"
                       "  image path  (.jpg/.png/…)\n"
                       "  video path  (.mp4/.avi/…)\n"
                       "  stream URL  (rtsp://…)\n"
                       "  webcam index (0, 1, …)"
                   ))
    p.add_argument(
        "--output",
        nargs="?",          # absent → None; --output alone → sentinel; --output path → path
        const="__auto__",   # sentinel: --output given with no value
        default=None,       # --output absent entirely
        metavar="PATH",
        help=(
            "Output path (optional):\n"
            "  omitted          → do not save output\n"
            "  --output         → auto-name: ./outputs/{source}_{driving}.png/mp4\n"
            "  --output PATH    → save to PATH"
        ),
    )
    p.add_argument("--model_name", default="LHM-1B",
                   choices=["LHM-500M","LHM-1B","LHM-500M-HF","LHM-1B-HF","LHM-MINI"])
    p.add_argument("--bg", default="white", choices=["white", "driving"],
                   help="'white' = plain bg; 'driving' = composite onto driving frame bg.")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--timer",
        nargs="?",            # absent → None; --timer alone → sentinel; --timer path → path
        const="__auto__",     # sentinel: --timer given with no value
        default=None,         # --timer absent entirely
        metavar="PATH",
        help=(
            "Timing report (optional):\n"
            "  omitted          → no timing calculated\n"
            "  --timer          → auto-name: ./outputs/{source}_{driving}_{type}_timings_NN.txt\n"
            "  --timer PATH     → write timings to PATH"
        ),
    )
    p.add_argument(
        "--stream",
        nargs="?",               # optional value: --stream  OR  --stream mjpeg
        const="mjpeg",           # default when flag given with no value
        default=None,            # default when flag is absent entirely
        metavar="MODE",
        choices=["mjpeg", "window", "rtsp"],
        help=(
            "Live preview mode (video/stream input only):\n"
            "  mjpeg   HTTP MJPEG on :8080 — open in any browser\n"
            "  window  cv2.imshow — needs a display (X11/VNC)\n"
            "  rtsp    push to RTSP via ffmpeg (needs rtsp-simple-server)\n"
            "Omit value to default to mjpeg."
        ),
    )
    p.add_argument(
        "--save-avatar",
        dest="save_avatar",
        default=True,
        type=lambda x: x.lower() not in ("false", "0", "no"),
        metavar="BOOL",
        help=(
            "Cache the built 3-D avatar under ./avatars/{source_name}/\n"
            "and reuse it automatically on subsequent runs (default: true).\n"
            "Pass --save-avatar false to disable caching."
        ),
    )
    return p.parse_args()


def main():
    args = _parse_args()

    # ── Resolve output path ───────────────────────────────────────────────────
    if args.output is None:
        print("[info] No --output given — output will not be saved.")
    elif args.output == "__auto__":
        src_stem    = os.path.splitext(os.path.basename(args.source))[0]
        drv_stem    = os.path.splitext(os.path.basename(args.driving))[0]
        _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        drv_ext     = os.path.splitext(args.driving)[-1].lower()
        out_ext     = ".png" if (drv_ext in _IMAGE_EXTS and os.path.isfile(args.driving)) else ".mp4"
        args.output = os.path.join("./outputs", f"{src_stem}_{drv_stem}{out_ext}")
        print(f"[info] --output auto-path: {args.output}")

    # ── Resolve timer path ────────────────────────────────────────────────────
    if args.timer is None:
        timer_path = None
    elif args.timer == "__auto__":
        src_stem    = os.path.splitext(os.path.basename(args.source))[0]
        drv_stem    = os.path.splitext(os.path.basename(args.driving))[0]
        _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        drv_ext     = os.path.splitext(args.driving)[-1].lower()
        drv_type    = "img" if (drv_ext in _IMAGE_EXTS and os.path.isfile(args.driving)) else "vid"
        _base       = f"{src_stem}_{drv_stem}_{drv_type}_timings"
        _next_n     = 99   # fallback: overwrite 99 if all slots taken
        for _n in range(1, 100):
            if not os.path.exists(os.path.join("./outputs", f"{_base}_{_n:02d}.txt")):
                _next_n = _n
                break
        timer_path  = os.path.join("./outputs", f"{_base}_{_next_n:02d}.txt")
        print(f"[info] --timer auto-path: {timer_path}")
    else:
        timer_path = args.timer

    run(
        source_path  = args.source,
        driving      = args.driving,
        output_path  = args.output,
        model_name   = args.model_name,
        bg_mode      = args.bg,
        device       = args.device,
        timer_path   = timer_path,
        stream_mode  = args.stream,
        save_avatar  = args.save_avatar,
    )


if __name__ == "__main__":
    main()