#!/usr/bin/env python3
"""
LHM Live Avatar Driver
----------------------
Builds or loads a cached 3D avatar from a source image, then animates
it using poses extracted from a driving video or webcam.

Avatar cache (.lhm_avatar) is stored in --avatar-dir (default: ./avatars/).

Usage:
    # Case 1 & 2: auto-detect or build avatar from source image
    python drive_avatar.py --source person.jpg --driving video.mp4

    # Case 3: provide avatar path directly (skip source image entirely)
    python drive_avatar.py --avatar avatars/person.lhm_avatar --driving video.mp4

    # Save output video
    python drive_avatar.py --source person.jpg --driving video.mp4 --output out.mp4

    # Stream live while processing
    python drive_avatar.py --source person.jpg --driving video.mp4 --stream

    # Webcam
    python drive_avatar.py --source person.jpg --driving webcam --stream
"""

import argparse
import os
import sys
import tempfile
import time

import cv2

# disable GUI if no display is available (headless container)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    _test = cv2.namedWindow('__test__', cv2.WINDOW_NORMAL)
    cv2.destroyWindow('__test__')
    _HAS_DISPLAY = True
except Exception:
    _HAS_DISPLAY = False

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from engine.pose_estimation.pose_estimator import PoseEstimator
from engine.SegmentAPI.base import Bbox
from LHM.utils.model_download_utils import AutoModelQuery
from LHM.utils.model_card import MODEL_CARD, MODEL_CONFIG
from LHM.utils.download_utils import download_extract_tar_from_url, download_from_url
from LHM.utils.face_detector import FaceDetector
from LHM.utils.hf_hub import wrap_model_hub
from LHM.runners.infer.human_lrm import (
    infer_preprocess_image,
    get_bbox,
    query_model_config,
)

try:
    from engine.SegmentAPI.SAM import SAM2Seg
    HAS_SAM = True
except Exception:
    print("\033[33m[WARN] SAM2 not found, falling back to rembg.\033[0m")
    from rembg import remove as rembg_remove
    HAS_SAM = False

ASPECT_STANDARD = 5.0 / 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32


# ── prior / geo file check ────────────────────────────────────────────────────

def prior_check():
    if not os.path.exists("./pretrained_models"):
        download_extract_tar_from_url(MODEL_CARD["prior_model"])
    geo_ply = "./pretrained_models/dense_sample_points/1_20000.ply"
    if not os.path.exists(geo_ply):
        download_from_url(
            "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com"
            "/share/aigc3d/data/LHM/1_20000.ply",
            "./pretrained_models/dense_sample_points/",
        )


# ── model loading ─────────────────────────────────────────────────────────────

def load_lhm_model(model_name: str):
    # accelerate must be initialized before any model logging calls
    from accelerate import PartialState
    PartialState()

    query      = AutoModelQuery()
    model_path = query.query(model_name)

    cfg_path   = query_model_config(model_name)
    cfg_train  = OmegaConf.load(cfg_path)

    cfg = OmegaConf.create()
    cfg.model_name  = model_path
    cfg.source_size = cfg_train.dataset.source_image_res
    try:
        cfg.src_head_size = cfg_train.dataset.src_head_size
    except Exception:
        cfg.src_head_size = 112
    cfg.render_size = cfg_train.dataset.render_image.high

    from LHM.models import model_dict
    hf_cls = wrap_model_hub(model_dict["human_lrm_sapdino_bh_sd3_5"])
    model  = hf_cls.from_pretrained(cfg.model_name)
    model.to(DEVICE, DTYPE).eval()
    return model, cfg


# ── avatar cache: save / load ─────────────────────────────────────────────────

def avatar_path_for_source(source_image_path: str, avatar_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(source_image_path))[0]
    return os.path.join(avatar_dir, f"{stem}.lhm_avatar")


def save_avatar(path: str, gs_model_list, query_points, transform_mat, shape_param):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "gs_model_list":  gs_model_list,
        "query_points":   query_points,
        "transform_mat":  transform_mat,
        "shape_param":    shape_param,
    }, path)
    print(f"[INFO] Avatar saved → {path}")


def load_avatar(path: str):
    print(f"[INFO] Loading avatar from {path}")
    data = torch.load(path, map_location=DEVICE)
    return (
        data["gs_model_list"],
        data["query_points"],
        data["transform_mat"],
        data["shape_param"],
    )


# ── build avatar from source image ───────────────────────────────────────────

def build_avatar(source_path, model, cfg, pose_estimator, face_detector, parsingnet):
    print(f"[INFO] Building avatar from: {source_path}")

    # segmentation
    if parsingnet is not None:
        out          = parsingnet(img_path=source_path, bbox=None)
        parsing_mask = (out.masks * 255).astype(np.uint8)
    else:
        img_np       = cv2.imread(source_path)
        remove_np    = rembg_remove(img_np)
        parsing_mask = remove_np[..., 3]

    # body shape from source
    shape_pose = pose_estimator(source_path)
    assert shape_pose.is_full_body, (
        f"Source image rejected: {shape_pose.msg}\nUse a full-body photo."
    )
    shape_param = torch.tensor(shape_pose.beta, dtype=DTYPE).unsqueeze(0)

    # preprocess source image
    image, _, _ = infer_preprocess_image(
        source_path,
        mask=parsing_mask,
        intr=None, pad_ratio=0, bg_color=1.0,
        max_tgt_size=896,
        aspect_standard=ASPECT_STANDARD,
        enlarge_ratio=[1.0, 1.0],
        render_tgt_size=cfg.source_size,
        multiply=14, need_mask=True,
    )

    # face crop
    try:
        rgb   = np.array(Image.open(source_path))[..., :3]
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
        bbox  = face_detector(rgb_t)
        head  = rgb_t[:, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        head  = cv2.resize(
            head.permute(1, 2, 0).cpu().numpy(),
            (cfg.src_head_size, cfg.src_head_size),
            interpolation=cv2.INTER_AREA,
        )
    except Exception:
        print("[WARN] Face detection failed; using blank head crop.")
        head = np.zeros((cfg.src_head_size, cfg.src_head_size, 3), dtype=np.uint8)

    src_head = (
        torch.from_numpy(head / 255.0).float().permute(2, 0, 1).unsqueeze(0)
    )

    # canonical zero-pose smplx for avatar init
    dummy = {
        "betas":      shape_param.to(DEVICE),
        "root_pose":  torch.zeros(1, 1, 3).to(DEVICE),
        "body_pose":  torch.zeros(1, 1, 21, 3).to(DEVICE),
        "jaw_pose":   torch.zeros(1, 1, 3).to(DEVICE),
        "leye_pose":  torch.zeros(1, 1, 3).to(DEVICE),
        "reye_pose":  torch.zeros(1, 1, 3).to(DEVICE),
        "lhand_pose": torch.zeros(1, 1, 15, 3).to(DEVICE),
        "rhand_pose": torch.zeros(1, 1, 15, 3).to(DEVICE),
        "expr":       torch.zeros(1, 1, 100).to(DEVICE),
        "trans":      torch.zeros(1, 1, 3).to(DEVICE),
    }

    with torch.no_grad():
        gs_model_list, query_points, transform_mat = model.infer_single_view(
            image.unsqueeze(0).to(DEVICE, DTYPE),
            src_head.unsqueeze(0).to(DEVICE, DTYPE),
            None, None,
            render_c2ws=None,
            render_intrs=None,
            render_bg_colors=None,
            smplx_params=dummy,
        )

    print("[INFO] Avatar built successfully.")
    return gs_model_list, query_points, transform_mat, shape_param


# ── pose extraction from a single driving frame ───────────────────────────────

def extract_pose_from_frame(frame_bgr, pose_estimator, tmp_dir, frame_idx):
    tmp_path = os.path.join(tmp_dir, f"frame_{frame_idx:06d}.jpg")
    cv2.imwrite(tmp_path, frame_bgr)
    try:
        result = pose_estimator(tmp_path)
    except Exception as e:
        print(f"[WARN] Pose estimation error on frame {frame_idx}: {e}")
        return None
    if not result.is_full_body:
        print(f"[WARN] Frame {frame_idx}: {result.msg}")
        return None

    h, w = frame_bgr.shape[:2]
    return {
        "betas":       torch.tensor(result.beta, dtype=DTYPE).unsqueeze(0),
        "root_pose":   torch.zeros(1, 3),
        "body_pose":   torch.zeros(1, 21, 3),
        "jaw_pose":    torch.zeros(1, 3),
        "leye_pose":   torch.zeros(1, 3),
        "reye_pose":   torch.zeros(1, 3),
        "lhand_pose":  torch.zeros(1, 15, 3),
        "rhand_pose":  torch.zeros(1, 15, 3),
        "expr":        torch.zeros(1, 100),
        "trans":       torch.zeros(1, 3),
        "focal":       torch.tensor([1000.0, 1000.0]),
        "princpt":     torch.tensor([w / 2.0, h / 2.0]),
        "img_size_wh": torch.tensor([float(w), float(h)]),
    }


def smplx_to_batch(smplx_param, shape_param, transform_mat):
    """Wrap single-frame params into [1,1,...] tensors for animation_infer."""
    b = {}
    b["betas"]                    = shape_param.to(DEVICE)
    b["transform_mat_neutral_pose"] = transform_mat
    for k in ["root_pose", "jaw_pose", "leye_pose", "reye_pose", "trans"]:
        b[k] = smplx_param[k].unsqueeze(0).to(DEVICE)
    b["body_pose"]   = smplx_param["body_pose"].unsqueeze(0).to(DEVICE)
    b["lhand_pose"]  = smplx_param["lhand_pose"].unsqueeze(0).to(DEVICE)
    b["rhand_pose"]  = smplx_param["rhand_pose"].unsqueeze(0).to(DEVICE)
    b["expr"]        = smplx_param["expr"].unsqueeze(0).to(DEVICE)
    b["focal"]       = smplx_param["focal"].unsqueeze(0).unsqueeze(0).to(DEVICE)
    b["princpt"]     = smplx_param["princpt"].unsqueeze(0).unsqueeze(0).to(DEVICE)
    b["img_size_wh"] = smplx_param["img_size_wh"].unsqueeze(0).unsqueeze(0).to(DEVICE)
    return b


def build_render_camera(smplx_param):
    focal, princpt = smplx_param["focal"], smplx_param["princpt"]
    intr = torch.eye(4)
    intr[0, 0], intr[1, 1] = focal[0], focal[1]
    intr[0, 2], intr[1, 2] = princpt[0], princpt[1]
    c2w = torch.eye(4)
    return (
        c2w.unsqueeze(0).unsqueeze(0).to(DEVICE),
        intr.unsqueeze(0).unsqueeze(0).to(DEVICE),
        torch.ones(1, 1, 3).to(DEVICE),
    )


# ── render one avatar frame ───────────────────────────────────────────────────

@torch.no_grad()
def render_frame(model, gs_model_list, query_points,
                 smplx_batch, render_c2ws, render_intrs, render_bg_colors):
    res       = model.animation_infer(
        gs_model_list, query_points, smplx_batch,
        render_c2ws=render_c2ws,
        render_intrs=render_intrs,
        render_bg_colors=render_bg_colors,
    )
    rgb  = res["comp_rgb"]
    mask = res["comp_mask"]
    mask[mask < 0.5] = 0.0
    out  = (rgb * mask + (1 - mask)).clamp(0, 1)
    out  = (out * 255).to(torch.uint8)[0].cpu().numpy()
    torch.cuda.empty_cache()
    return out  # [H, W, 3]  RGB


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    prior_check()
    if not _HAS_DISPLAY and args.stream:
        print('[WARN] No display detected (headless container). --stream will be ignored. Use --output to save video.')


    # normalize all paths to absolute
    if args.source:
        args.source = os.path.abspath(args.source)
    if args.avatar:
        args.avatar = os.path.abspath(args.avatar)
    if args.driving and args.driving != 'webcam':
        args.driving = os.path.abspath(args.driving)
    if args.output:
        args.output = os.path.abspath(args.output)
    args.avatar_dir = os.path.abspath(args.avatar_dir)

    # ── resolve avatar (Cases 1 / 2 / 3) ─────────────────────────────────────
    if args.avatar:
        # Case 3: explicit avatar path provided
        if not os.path.exists(args.avatar):
            print(f"[ERROR] Avatar file not found: {args.avatar}")
            sys.exit(1)
        gs_model_list, query_points, transform_mat, shape_param = load_avatar(args.avatar)
        model, cfg = load_lhm_model(args.model)

    else:
        # Cases 1 & 2: derive avatar path from source image name
        if not args.source:
            print("[ERROR] Provide --source <image> or --avatar <path>.")
            sys.exit(1)

        avatar_path = avatar_path_for_source(args.source, args.avatar_dir)

        model, cfg = load_lhm_model(args.model)

        if os.path.exists(avatar_path):
            # Case 1: cached avatar found
            print(f"[INFO] Found cached avatar: {avatar_path}")
            gs_model_list, query_points, transform_mat, shape_param = load_avatar(avatar_path)
        else:
            # Case 2: build and cache
            print(f"[INFO] No cached avatar found at {avatar_path}. Building...")
            pose_estimator = PoseEstimator(
                "./pretrained_models/human_model_files/", device=DEVICE
            )
            face_detector = FaceDetector(
                "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
                device=DEVICE,
            )
            parsingnet = SAM2Seg() if HAS_SAM else None

            gs_model_list, query_points, transform_mat, shape_param = build_avatar(
                args.source, model, cfg, pose_estimator, face_detector, parsingnet
            )
            save_avatar(avatar_path, gs_model_list, query_points, transform_mat, shape_param)

    # ── open driving source ───────────────────────────────────────────────────
    pose_estimator = PoseEstimator(
        "./pretrained_models/human_model_files/", device=DEVICE
    )

    if args.driving == "webcam":
        cap = cv2.VideoCapture(0)
        print("[INFO] Opened webcam.")
    else:
        cap = cv2.VideoCapture(args.driving)
        print(f"[INFO] Opened driving video: {args.driving}")

    if not cap.isOpened():
        print("[ERROR] Could not open driving source.")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Driving source: {src_w}x{src_h} @ {src_fps:.1f} fps")

    # ── output video writer (lazy init on first frame) ────────────────────────
    writer      = None
    output_path = None
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        if args.driving == "webcam":
            fname = f"webcam_{int(time.time())}.mp4"
        else:
            fname = os.path.splitext(os.path.basename(args.driving))[0] + "_avatar.mp4"
        output_path = os.path.join(args.output, fname)
        print(f"[INFO] Output will be saved to: {output_path}")

    # ── per-frame loop ────────────────────────────────────────────────────────
    frame_idx  = 0
    last_smplx = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("[INFO] Processing frames. Press Q to quit live view.")

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            t0 = time.time()

            smplx_param = extract_pose_from_frame(
                frame_bgr, pose_estimator, tmp_dir, frame_idx
            )
            if smplx_param is None:
                if last_smplx is None:
                    frame_idx += 1
                    continue
                smplx_param = last_smplx
            else:
                last_smplx = smplx_param

            smplx_batch                     = smplx_to_batch(smplx_param, shape_param, transform_mat)
            render_c2ws, render_intrs, bgs  = build_render_camera(smplx_param)

            try:
                rendered = render_frame(
                    model, gs_model_list, query_points,
                    smplx_batch, render_c2ws, render_intrs, bgs,
                )
            except Exception as e:
                print(f"[WARN] Render failed frame {frame_idx}: {e}")
                frame_idx += 1
                continue

            elapsed = time.time() - t0
            print(f"[frame {frame_idx:05d}]  {elapsed*1000:.0f} ms  ({1/elapsed:.1f} fps)")

            # write to file
            if output_path is not None:
                h, w = rendered.shape[:2]
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))
                writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))

            # live display
            if (args.stream or args.driving == "webcam") and _HAS_DISPLAY:
                h, w = rendered.shape[:2]
                driving_resized = cv2.resize(frame_bgr, (w, h))
                side_by_side    = np.concatenate(
                    [driving_resized, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)],
                    axis=1,
                )
                cv2.imshow("LHM  |  driving (left)  avatar (right)  |  Q to quit",
                           side_by_side)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] User quit.")
                    break
            elif args.stream and not _HAS_DISPLAY:
                print("[WARN] --stream requested but no display available (headless). Skipping preview.", flush=True)
                args.stream = False  # suppress repeated warnings

            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[INFO] Saved {frame_idx} frames → {output_path}")
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LHM Live Avatar Driver")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--source", default=None,
                     help="Source image path (Cases 1 & 2)")
    src.add_argument("--avatar", default=None,
                     help="Path to a pre-built .lhm_avatar file (Case 3)")

    p.add_argument("--driving",    required=True,
                   help="Driving video path, or 'webcam'")
    p.add_argument("--output",     default=None,
                   help="Directory to save output video. Omit to skip saving.")
    p.add_argument("--avatar-dir", default="./avatars",
                   help="Directory for cached avatars (default: ./avatars)")
    p.add_argument("--model",      default="LHM-1B",
                   choices=["LHM-MINI", "LHM-500M", "LHM-500M-HF",
                             "LHM-1B",  "LHM-1B-HF"],
                   help="LHM model to use (default: LHM-1B)")
    p.add_argument("--stream",     action="store_true",
                   help="Show live preview window while processing")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())