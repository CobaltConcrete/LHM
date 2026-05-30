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
    [--device cuda]
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

def _extract_driving_pose(
    driving_path: str,
    pose_estimator: PoseEstimator,
    device: str = "cuda",
) -> tuple:
    """
    Run multiHMR on the driving image and build smplx_params + camera tensors
    in the exact format used by the pre-extracted smplx_params/*.json files.

    Confirmed JSON format (from an actual pre-extracted file):
        root_pose   [3]        axis-angle, ~[π,0,0] = face-forward in model space
        body_pose   [21,3]     per-joint axis-angle
        jaw_pose    [3]
        leye_pose   [3]
        reye_pose   [3]
        lhand_pose  [15,3]
        rhand_pose  [15,3]
        trans       [3]        metric, z≈5-6 (distance from camera)
        betas       [10]
        expr        [100]
        focal       [2]        pixel-space focal lengths of the *padded* image
        princpt     [2]        pixel-space principal point of the *padded* image
        img_size_wh [2]        width, height of the *padded* image

    The critical coordinate-frame fix
    ----------------------------------
    multiHMR `global_orient` is in OpenCV camera space (x right, y down, z into
    screen).  The pre-extracted jsons store `root_pose` already rotated so the
    person faces forward in the model's canonical space (Y-up, Z toward viewer).
    The transform between the two is a ~π rotation around X:

        R_canon = R_x(π) @ R_camera

    In axis-angle:  R_x(π) = [π, 0, 0]
    We compose using pytorch3d's axis_angle_to_matrix / matrix_to_axis_angle.
    """
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

    # ── 1. Run multiHMR ───────────────────────────────────────────────────────
    img_np_raw                    = np.asarray(Image.open(driving_path).convert("RGB"))
    img_np_pad, offset_w, offset_h = pose_estimator.img_center_padding(img_np_raw)
    img_tensor, annotation        = pose_estimator._preprocess(img_np_pad)
    K_cam                         = pose_estimator.get_camera_parameters()  # [1,3,3]

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            target_humans = pose_estimator.mhmr_model(
                img_tensor,
                is_training=False,
                nms_kernel_size=3,
                det_thresh=0.3,
                K=K_cam,
                idx=None,
                max_dist=None,
            )

    if len(target_humans) == 0:
        raise RuntimeError("No person detected in the driving image.")
    if len(target_humans) > 1:
        print(f"[warn] {len(target_humans)} people detected – using first.")

    human = target_humans[0]

    # ── Full structure dump ───────────────────────────────────────────────────
    def _dump(d, indent=0):
        prefix = "  " * indent
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                print(f"{prefix}{k}: Tensor {list(v.shape)} dtype={v.dtype}")
            elif isinstance(v, dict):
                print(f"{prefix}{k}: dict")
                _dump(v, indent + 1)
            elif isinstance(v, np.ndarray):
                print(f"{prefix}{k}: ndarray {v.shape} dtype={v.dtype}")
            else:
                print(f"{prefix}{k}: {type(v).__name__} = {v}")
    print("\n[info] Full multiHMR human[0] structure:")
    _dump(human)
    print()

    # ── 2. Unpack rotvec [53, 3] ─────────────────────────────────────────────
    # multiHMR packs all joint rotations into a single rotvec tensor [53, 3]:
    #   [0]     global_orient / root_pose
    #   [1:22]  body_pose  (21 joints)
    #   [22:37] left_hand_pose (15 joints)
    #   [37:52] right_hand_pose (15 joints)
    #   [52]    jaw_pose
    # leye_pose / reye_pose are not in rotvec → use zeros
    rotvec = human['rotvec'].float().cpu().detach()  # [53, 3]

    root_pose_cam = rotvec[0]         # [3]   camera space – corrected below
    body_pose     = rotvec[1:22]      # [21, 3]
    lhand_pose    = rotvec[22:37]     # [15, 3]
    rhand_pose    = rotvec[37:52]     # [15, 3]
    jaw_pose      = rotvec[52]        # [3]
    leye_pose     = torch.zeros(3)
    reye_pose     = torch.zeros(3)

    # expression: multiHMR outputs [10], JSON expects [100] → pad with zeros
    expr_raw = human['expression'].float().cpu().detach()  # [10]
    expr     = torch.zeros(100)
    expr[:expr_raw.shape[0]] = expr_raw

    trans = human['transl'].float().cpu().detach()         # [3]
    betas = human['shape'].float().cpu().detach()          # [10]

    # ── 4. Fix coordinate frame: camera space → model canonical space ─────────
    # Pre-extracted JSONs have root_pose x-component ≈ +π, meaning the
    # extraction pipeline post-multiplied R_x(π) onto global_orient:
    #   R_stored = R_mhmr @ R_x(π)
    # So we do the same here.
    R_x_pi = torch.zeros(3, dtype=torch.float32)
    R_x_pi[0] = torch.pi                                            # [π, 0, 0]
    R_x_pi_mat  = axis_angle_to_matrix(R_x_pi.unsqueeze(0))        # [1,3,3]
    R_cam_mat   = axis_angle_to_matrix(root_pose_cam.unsqueeze(0))  # [1,3,3]
    R_canon_mat = R_cam_mat @ R_x_pi_mat                            # [1,3,3]
    root_pose   = matrix_to_axis_angle(R_canon_mat)[0]              # [3]

    # ── 5. Camera intrinsics in padded-image pixel space ──────────────────────
    # pose_estimator runs on a square 896×896 crop; the pre-extracted JSONs
    # store focal/princpt in the original *padded* image space (before the
    # 896-resize), so we must back-project through the annotation transform.
    pad_left, pad_top, scale_factor, _, _ = annotation
    pad_h, pad_w = img_np_pad.shape[:2]

    # The 896-px K was computed for the resized image; undo the scale to get
    # back to padded-image pixel coords, then subtract the padding offset.
    K33 = K_cam[0].cpu()                        # [3,3], in 896-px space
    fx  = K33[0,0].item() / scale_factor
    fy  = K33[1,1].item() / scale_factor
    cx  = (K33[0,2].item() - pad_left) / scale_factor
    cy  = (K33[1,2].item() - pad_top)  / scale_factor

    focal_np   = np.array([fx, fy],       dtype=np.float32)
    princpt_np = np.array([cx, cy],       dtype=np.float32)
    img_wh_np  = np.array([pad_w, pad_h], dtype=np.float32)  # padded image size

    # ── 6. Pack into [1,1,...] tensors matching prepare_motion_seqs output ────
    def _bf(t):
        return t.unsqueeze(0).unsqueeze(0).to(device)

    smplx_params = {
        "betas":       betas.unsqueeze(0).to(device),   # [1,10]
        "root_pose":   _bf(root_pose),                  # [1,1,3]
        "body_pose":   _bf(body_pose),                  # [1,1,21,3]
        "jaw_pose":    _bf(jaw_pose),                   # [1,1,3]
        "leye_pose":   _bf(leye_pose),                  # [1,1,3]
        "reye_pose":   _bf(reye_pose),                  # [1,1,3]
        "lhand_pose":  _bf(lhand_pose),                 # [1,1,15,3]
        "rhand_pose":  _bf(rhand_pose),                 # [1,1,15,3]
        "expr":        _bf(expr),                       # [1,1,100]
        "trans":       _bf(trans),                      # [1,1,3]
        "focal":       _bf(torch.from_numpy(focal_np)), # [1,1,2]
        "princpt":     _bf(torch.from_numpy(princpt_np)),# [1,1,2]
        "img_size_wh": _bf(torch.from_numpy(img_wh_np)),# [1,1,2]
    }

    # ── 7. Render camera: identity c2w (same as _load_pose in utils.py) ───────
    intr_4x4         = _build_intrinsic_4x4(focal_np, princpt_np)
    render_c2ws      = torch.eye(4).unsqueeze(0).unsqueeze(0).to(device)   # [1,1,4,4]
    render_intrs     = intr_4x4.unsqueeze(0).unsqueeze(0).to(device)       # [1,1,4,4]
    render_bg_colors = torch.ones(1, 1, 3, dtype=torch.float32, device=device)


    # ── DEBUG: print full smplx_params as JSON for comparison ────────────────
    import json as _json
    _debug = {k: v.cpu().squeeze().tolist() for k, v in smplx_params.items()}
    print("\n========== SMPLX PARAMS DEBUG ==========")
    print(_json.dumps(_debug, indent=2))
    print("=========================================\n")
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
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(
    source_path:  str,
    driving_path: str,
    output_path:  str,
    model_name:   str  = "LHM-1B",
    bg_mode:      str  = "white",
    device:       str  = "cuda",
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ── 0. Downloads ─────────────────────────────────────────────────────────
    _download_geo_files()
    _prior_check()

    # ── 1. Auto-select model ─────────────────────────────────────────────────
    switcher   = AutoModelSwitcher(MEMORY_MODEL_CARD, extra_memory=0)
    model_name = switcher.query(model_name)
    print(f"[info] Model: {model_name}")

    os.environ.update({
        "APP_ENABLED":    "1",
        "APP_MODEL_NAME":  model_name,
        "APP_TYPE":       "infer.human_lrm",
        "NUMBA_THREADING_LAYER": "omp",
    })

    # ── 2. Load all sub-models ───────────────────────────────────────────────
    print("[info] Loading pose estimator …")
    pose_estimator = PoseEstimator(
        "./pretrained_models/human_model_files/", device="cpu"
    )
    pose_estimator.to(device)
    pose_estimator.device = device

    print("[info] Loading face detector …")
    # human_lrm.py uses FaceDetector (not VGGHeadDetector from app.py)
    face_detector = FaceDetector(
        "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
        device=device,
    )

    print("[info] Loading segmentation …")
    parsing_net = SAM2Seg() if _HAS_SAM2 else None

    cfg, _ = _parse_configs(model_name)

    print("[info] Loading LHM …")
    lhm = _build_model(cfg)
    lhm.to(device)
    lhm.eval()

    source_size   = cfg.source_size
    render_size   = cfg.render_size
    src_head_size = cfg.src_head_size
    dtype         = torch.float32

    # ── 3. Preprocess source image ───────────────────────────────────────────
    print("[info] Preprocessing source image …")
    src_mask = _segment(source_path, parsing_net)

    src_rgb, _ = _infer_preprocess_image(
        source_path, src_mask,
        max_tgt_size=896,
        aspect_standard=ASPECT_STANDARD,
        render_tgt_size=source_size,
        bg_color=1.0,
        multiply=14,
    )
    src_head = _crop_face(source_path, face_detector, src_head_size)

    # ── 4. Extract driving pose ──────────────────────────────────────────────
    print("[info] Extracting driving pose …")
    smplx_params, render_c2ws, render_intrs, render_bg_colors = (
        _extract_driving_pose(driving_path, pose_estimator, device=device)
    )

    # ── 5. Override betas with SOURCE person's shape from PoseEstimator ──────
    # The driving image gives us pose; the source image gives us body shape.
    print("[info] Estimating source body shape …")
    src_shape = pose_estimator(source_path)
    if src_shape.is_full_body and src_shape.beta is not None:
        smplx_params["betas"] = (
            torch.tensor(src_shape.beta, dtype=dtype).unsqueeze(0).to(device)
        )
        print("[info] Using source person's body shape (beta).")
    else:
        print("[warn] Could not estimate source body shape – using driving person's betas.")

    # ── 6. Build avatar ──────────────────────────────────────────────────────
    print("[info] Building 3-D avatar …")
    lhm.to(dtype)
    t0 = time.time()

    with torch.no_grad():
        gs_model_list, query_points, transform_mat_neutral_pose = lhm.infer_single_view(
            src_rgb.unsqueeze(0).to(device, dtype),
            src_head.unsqueeze(0).to(device, dtype),
            None, None,
            render_c2ws=render_c2ws,
            render_intrs=render_intrs,
            render_bg_colors=render_bg_colors,
            smplx_params={k: v.to(device) for k, v in smplx_params.items()},
        )
    print(f"[info] Avatar built in {time.time()-t0:.2f}s")

    # ── 7. Render in driving pose ────────────────────────────────────────────
    print("[info] Rendering …")
    smplx_params["transform_mat_neutral_pose"] = transform_mat_neutral_pose

    with torch.no_grad():
        res = lhm.animation_infer(
            gs_model_list, query_points, smplx_params,
            render_c2ws=render_c2ws,
            render_intrs=render_intrs,
            render_bg_colors=render_bg_colors,
        )

    comp_rgb  = res["comp_rgb"]   # [1, H, W, 3]
    comp_mask = res["comp_mask"]  # [1, H, W, 1 or 3]
    comp_mask[comp_mask < 0.5] = 0.0

    rendered_np = (
        (comp_rgb * comp_mask + (1 - comp_mask) * 1.0)
        .clamp(0, 1)[0].cpu().numpy() * 255
    ).astype(np.uint8)

    # ── 8. Composite ─────────────────────────────────────────────────────────
    if bg_mode == "driving":
        print("[info] Compositing onto driving background …")
        drv_np   = np.array(Image.open(driving_path).convert("RGB"))
        dh, dw   = drv_np.shape[:2]
        rend_res = cv2.resize(rendered_np, (dw, dh), interpolation=cv2.INTER_LANCZOS4)
        mask_np  = comp_mask[0].clamp(0,1).cpu().numpy()
        if mask_np.ndim == 3 and mask_np.shape[-1] != 1:
            mask_np = mask_np[..., :1]
        mask_res = cv2.resize(
            mask_np.squeeze(-1) if mask_np.ndim==3 else mask_np,
            (dw, dh), interpolation=cv2.INTER_LANCZOS4
        )[:,:,None]
        output_img = (
            rend_res * mask_res + drv_np * (1 - mask_res)
        ).clip(0, 255).astype(np.uint8)
    else:
        output_img = rendered_np

    # ── 9. Save ──────────────────────────────────────────────────────────────
    Image.fromarray(output_img).save(output_path)
    print(f"[✓] Saved → {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="LHM retargeting: transfer source identity to driving pose."
    )
    p.add_argument("--source",  required=True,
                   help="Source image path (identity donor).")
    p.add_argument("--driving", required=True,
                   help="Driving image path (pose donor).")
    p.add_argument("--output",  required=True,
                   help="Output path (.png) or directory.")
    p.add_argument("--model_name", default="LHM-1B",
                   choices=["LHM-500M","LHM-1B","LHM-500M-HF","LHM-1B-HF","LHM-MINI"])
    p.add_argument("--bg", default="white", choices=["white","driving"],
                   help="'white' = plain bg; 'driving' = paste onto driving image bg.")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = _parse_args()
    out  = args.output
    if out.endswith("/") or os.path.isdir(out):
        out = os.path.join(out, "result.png")
    run(
        source_path  = args.source,
        driving_path = args.driving,
        output_path  = out,
        model_name   = args.model_name,
        bg_mode      = args.bg,
        device       = args.device,
    )


if __name__ == "__main__":
    main()