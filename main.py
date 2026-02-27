# ============================================================
# Bubble 3D reconstruction (NO STICKING, NO CENTER STRIP, LESS TAILS)
# - per-bubble reconstruction (connected components + matching by X overlap)
# - NO interpolation across gaps (fill only inside valid segments)
# - robust column profile (percentiles instead of min/max) -> fewer "tails"
# - live preview: animates, then keeps LAST FRAME until you close the window
# - start_frame (1-based) to choose where to start in sorted file_name list
# ============================================================

import json
import os
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import pyvista as pv

# (optional) SciPy smoothing
try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ==========================================
# STABILNA WIZUALIZACJA (zakomentowana)
# ==========================================
def show_step(img, title="", cmap=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    plt.waitforbuttonpress()
    plt.close(fig)


# ==========================================
# COCO LOADING
# ==========================================
def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    return coco


# ==========================================
# MASKA Z POLYGONU LUB BBOX
# ==========================================
def ann_to_mask(ann, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    seg = ann.get("segmentation", None)

    if isinstance(seg, list) and len(seg) > 0:
        for poly in seg:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    bbox = ann.get("bbox", None)
    if bbox is None or len(bbox) != 4:
        return mask

    x, y, w, h = map(float, bbox)
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(img_w, int(round(x + w)))
    y1 = min(img_h, int(round(y + h)))
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask


# ==========================================
# RURKI = DWIE LINIE y = ax + b
# ==========================================
def point_in_tube(xc: float, yc: float, tube: dict, margin_px: float = 0.0) -> bool:
    y_top = tube["a_top"] * xc + tube["b_top"]
    y_bot = tube["a_bot"] * xc + tube["b_bot"]
    y_min = min(y_top, y_bot)
    y_max = max(y_top, y_bot)
    return (y_min + margin_px) <= yc <= (y_max - margin_px)


def draw_tubes(rgb, tubes):
    out = rgb.copy()
    h, w, _ = out.shape

    for i, t in enumerate(tubes):
        x0, x1 = 0, w - 1
        y0_top = int(round(t["a_top"] * x0 + t["b_top"]))
        y1_top = int(round(t["a_top"] * x1 + t["b_top"]))
        y0_bot = int(round(t["a_bot"] * x0 + t["b_bot"]))
        y1_bot = int(round(t["a_bot"] * x1 + t["b_bot"]))

        cv2.line(out, (x0, y0_top), (x1, y1_top), (255, 255, 0), 2)
        cv2.line(out, (x0, y0_bot), (x1, y1_bot), (255, 255, 0), 2)

        y_mid = int((y0_top + y0_bot) / 2)
        cv2.putText(out, f"tube {i+1}", (10, max(20, y_mid)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return out


def overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.4):
    out = rgb.copy()
    colored = np.zeros_like(out)
    colored[:, :] = color
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + colored[m] * alpha).astype(np.uint8)
    return out


def build_bubble_mask_from_anns(anns, img_h, img_w):
    m = np.zeros((img_h, img_w), dtype=np.uint8)
    for ann in anns:
        m = cv2.bitwise_or(m, ann_to_mask(ann, img_h, img_w))
    return m


# ==========================================
# WYCIĘCIE RURKI JAKO RÓWNOLEGŁOBOK + WYPROSTOWANIE
# ==========================================
def _tube_quad_points(tube: dict, x_left: float, x_right: float, margin_px: float = 0.0):
    a_top, b_top = float(tube["a_top"]), float(tube["b_top"])
    a_bot, b_bot = float(tube["a_bot"]), float(tube["b_bot"])

    y_tl = a_top * x_left + b_top + margin_px
    y_tr = a_top * x_right + b_top + margin_px
    y_bl = a_bot * x_left + b_bot - margin_px
    y_br = a_bot * x_right + b_bot - margin_px

    if y_bl < y_tl:
        y_tl, y_bl = y_bl, y_tl
    if y_br < y_tr:
        y_tr, y_br = y_br, y_tr

    return np.array([
        [x_left,  y_tl],
        [x_right, y_tr],
        [x_right, y_br],
        [x_left,  y_bl],
    ], dtype=np.float32)


def crop_and_rectify_tube(img, tube: dict, x_left: int, x_right: int,
                          inner_margin_px: float = 0.0,
                          out_width: int = None,
                          interp=cv2.INTER_LINEAR):
    x_left_f = float(x_left)
    x_right_f = float(x_right)

    src = _tube_quad_points(tube, x_left_f, x_right_f, margin_px=inner_margin_px)

    a_top, b_top = float(tube["a_top"]), float(tube["b_top"])
    a_bot, b_bot = float(tube["a_bot"]), float(tube["b_bot"])

    thick_left = abs((a_bot * x_left_f + b_bot) - (a_top * x_left_f + b_top)) - 2.0 * inner_margin_px
    thick_right = abs((a_bot * x_right_f + b_bot) - (a_top * x_right_f + b_top)) - 2.0 * inner_margin_px
    out_h = int(round(max(1.0, 0.5 * (thick_left + thick_right))))

    if out_width is None:
        out_w = int(round(max(2.0, x_right_f - x_left_f)))
    else:
        out_w = int(out_width)

    dst = np.array([
        [0,         0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0,         out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)

    border = cv2.BORDER_CONSTANT
    border_val = 0 if img.ndim == 2 else (0, 0, 0)

    rect = cv2.warpPerspective(img, M, (out_w, out_h),
                               flags=interp, borderMode=border, borderValue=border_val)
    return rect, src


# ==========================================
# SKALOWANIE / DOPASOWANIE
# ==========================================
def resize_to_match_height(img, target_h, keep_aspect=True, is_mask=False):
    h, w = img.shape[:2]
    if h == target_h:
        return img

    scale = target_h / float(h)
    if keep_aspect:
        new_w = max(1, int(round(w * scale)))
        new_h = target_h
    else:
        new_w = w
        new_h = target_h

    if is_mask:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA

    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def center_crop_or_pad_width(img, target_w, pad_value=0):
    h, w = img.shape[:2]
    if w == target_w:
        return img

    if w > target_w:
        x0 = (w - target_w) // 2
        return img[:, x0:x0 + target_w].copy()

    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    if img.ndim == 2:
        return cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right,
                                  borderType=cv2.BORDER_CONSTANT, value=pad_value)
    else:
        return cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right,
                                  borderType=cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))


def rectify_and_align_pair(mask_top_orig: np.ndarray, tube_top: dict,
                           mask_side_orig: np.ndarray, tube_side: dict,
                           x_left: int, x_right: int, inner_margin_px: float,
                           keep_aspect: bool):
    rect_top, quad_top = crop_and_rectify_tube(
        mask_top_orig, tube_top, x_left=x_left, x_right=x_right,
        inner_margin_px=inner_margin_px, interp=cv2.INTER_NEAREST
    )
    rect_side, quad_side = crop_and_rectify_tube(
        mask_side_orig, tube_side, x_left=x_left, x_right=x_right,
        inner_margin_px=inner_margin_px, interp=cv2.INTER_NEAREST
    )

    H_target = max(rect_top.shape[0], rect_side.shape[0])
    rect_top = resize_to_match_height(rect_top, H_target, keep_aspect=keep_aspect, is_mask=True)
    rect_side = resize_to_match_height(rect_side, H_target, keep_aspect=keep_aspect, is_mask=True)

    W_target = min(rect_top.shape[1], rect_side.shape[1])
    rect_top = center_crop_or_pad_width(rect_top, W_target, pad_value=0)
    rect_side = center_crop_or_pad_width(rect_side, W_target, pad_value=0)

    return rect_top, rect_side, quad_top, quad_side


# ==========================================
# CONNECTED COMPONENTS + MATCHING BY X-OVERLAP
# (rekonstrukcja per-bąbel, brak sklejania)
# ==========================================
def connected_components(mask_2d: np.ndarray, min_area: int = 80):
    m = (mask_2d > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        comps.append(comp)
    return comps


def x_span(mask_2d: np.ndarray):
    xs = np.where(mask_2d.max(axis=0) > 0)[0]
    if xs.size == 0:
        return None
    return int(xs.min()), int(xs.max())


def span_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax0, ax1 = a
    bx0, bx1 = b
    inter = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
    union = (ax1 - ax0 + 1) + (bx1 - bx0 + 1) - inter
    return inter / union if union > 0 else 0.0


def match_components_by_x(top_comps, side_comps, iou_thr=0.15):
    top_spans = [x_span(c) for c in top_comps]
    side_spans = [x_span(c) for c in side_comps]

    pairs = []
    used_side = set()
    for i, ts in enumerate(top_spans):
        best_j = -1
        best_iou = 0.0
        for j, ss in enumerate(side_spans):
            if j in used_side:
                continue
            iou = span_iou(ts, ss)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            used_side.add(best_j)
            pairs.append((top_comps[i], side_comps[best_j], best_iou))
    return pairs


# ============================================================
# Interpolacja TYLKO w obrębie ciągłych segmentów valid
# (usuwa pasek przez środek rurki / mostkowanie dziur)
# ============================================================
def _segments_from_valid(valid: np.ndarray, min_len: int = 3):
    segs = []
    n = valid.size
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        if (j - i) >= min_len:
            segs.append((i, j))
        i = j
    return segs


def _fill_missing_1d_within_segments(arr: np.ndarray, valid: np.ndarray):
    out = arr.astype(np.float32).copy()
    x = np.arange(arr.size)

    segs = _segments_from_valid(valid, min_len=2)
    for l, r in segs:
        v = valid[l:r]
        if v.sum() == 0:
            continue
        xx = x[l:r]
        yy = out[l:r]
        yy[~v] = np.interp(xx[~v], xx[v], yy[v])
        out[l:r] = yy
    return out


# ============================================================
# build 3D volume from TWO silhouettes using ELLIPTIC cross-sections
# - robust profile (percentiles) => mniej ogonów
# - fill only within valid segments => brak paska
# ============================================================
def _profile_from_mask_columns(mask_2d: np.ndarray, p_lo=5, p_hi=95):
    """
    mask_2d: uint8 [H,W], rows=vertical axis
    returns: center[W], half[W], valid[W]
    robust: uses percentiles instead of min/max (less tails)
    """
    H, W = mask_2d.shape
    center = np.zeros(W, dtype=np.float32)
    half = np.zeros(W, dtype=np.float32)
    valid = np.zeros(W, dtype=bool)

    for x in range(W):
        ys = np.where(mask_2d[:, x] > 0)[0]
        if ys.size == 0:
            center[x] = (H - 1) / 2.0
            half[x] = 0.0
            valid[x] = False
            continue

        # robust bounds
        y0 = int(np.percentile(ys, p_lo))
        y1 = int(np.percentile(ys, p_hi))
        if y1 < y0:
            y0, y1 = y1, y0

        center[x] = 0.5 * (y0 + y1)
        half[x] = 0.5 * (y1 - y0)
        valid[x] = True

    return center, half, valid


def build_volume_elliptic_from_two_masks(mask_top_xz: np.ndarray,
                                        mask_side_xy: np.ndarray,
                                        diameter_mm: float,
                                        voxel_mm: float = None,
                                        smooth_sigma_x: float = 2.0,
                                        min_radius_vox: float = 0.8):
    if mask_top_xz.ndim != 2 or mask_side_xy.ndim != 2:
        raise ValueError("maski muszą być 2D")
    H, W = mask_top_xz.shape
    if mask_side_xy.shape != (H, W):
        raise ValueError("mask_top i mask_side muszą mieć ten sam rozmiar po pipeline")

    mm_per_px = diameter_mm / float(H)
    if voxel_mm is None:
        voxel_mm = mm_per_px

    R_vox = int(round((diameter_mm / 2.0) / voxel_mm))
    YZ = 2 * R_vox + 1
    X = max(1, int(round((W * mm_per_px) / voxel_mm)))
    x_map = np.linspace(0, X - 1, W).round().astype(int)

    cy = R_vox
    cz = R_vox

    # TOP -> Z profile (rows are Z)
    zc_px, zhalf_px, zvalid = _profile_from_mask_columns(mask_top_xz)
    # SIDE -> Y profile (rows are Y)
    yc_px, yhalf_px, yvalid = _profile_from_mask_columns(mask_side_xy)

    valid_both = zvalid & yvalid

    # Fill only INSIDE valid segments (no bridging between bubbles)
    zc_px    = _fill_missing_1d_within_segments(zc_px, valid_both)
    yc_px    = _fill_missing_1d_within_segments(yc_px, valid_both)
    zhalf_px = _fill_missing_1d_within_segments(zhalf_px, valid_both)
    yhalf_px = _fill_missing_1d_within_segments(yhalf_px, valid_both)

    # Critical: outside valid -> radii 0 => no fill => no center strip
    zhalf_px[~valid_both] = 0.0
    yhalf_px[~valid_both] = 0.0

    px_center = (H - 1) / 2.0
    z0_mm = (zc_px - px_center) * mm_per_px
    y0_mm = (yc_px - px_center) * mm_per_px

    z0_vox = np.round(z0_mm / voxel_mm).astype(np.int32) + cz
    y0_vox = np.round(y0_mm / voxel_mm).astype(np.int32) + cy

    b_vox = np.maximum(0.0, (zhalf_px * mm_per_px) / voxel_mm).astype(np.float32)  # Z semi-axis
    a_vox = np.maximum(0.0, (yhalf_px * mm_per_px) / voxel_mm).astype(np.float32)  # Y semi-axis

    # Smooth only where we have data; simplest: smooth whole arrays but radii are 0 outside valid
    if smooth_sigma_x and smooth_sigma_x > 0:
        if HAS_SCIPY:
            a_vox = gaussian_filter1d(a_vox, sigma=smooth_sigma_x).astype(np.float32)
            b_vox = gaussian_filter1d(b_vox, sigma=smooth_sigma_x).astype(np.float32)
            y0_vox = np.round(gaussian_filter1d(y0_vox.astype(np.float32), sigma=smooth_sigma_x)).astype(np.int32)
            z0_vox = np.round(gaussian_filter1d(z0_vox.astype(np.float32), sigma=smooth_sigma_x)).astype(np.int32)
        else:
            k = int(max(3, 2 * round(smooth_sigma_x) + 1))
            k = k if (k % 2 == 1) else k + 1
            ker = np.ones(k, dtype=np.float32) / k
            a_vox = np.convolve(a_vox, ker, mode="same").astype(np.float32)
            b_vox = np.convolve(b_vox, ker, mode="same").astype(np.float32)

    # apply min radius only where we have non-zero after smoothing
    a_vox = np.where(a_vox > 0.0, np.maximum(a_vox, min_radius_vox), 0.0).astype(np.float32)
    b_vox = np.where(b_vox > 0.0, np.maximum(b_vox, min_radius_vox), 0.0).astype(np.float32)

    vol = np.zeros((X, YZ, YZ), dtype=bool)

    yy, zz = np.meshgrid(np.arange(YZ), np.arange(YZ), indexing="ij")
    circle = ((yy - cy) ** 2 + (zz - cz) ** 2) <= (R_vox ** 2)

    for x_px in range(W):
        # if no object here -> skip
        if a_vox[x_px] <= 0.0 or b_vox[x_px] <= 0.0:
            continue

        xv = int(x_map[x_px])

        y0 = int(np.clip(y0_vox[x_px], 0, YZ - 1))
        z0 = int(np.clip(z0_vox[x_px], 0, YZ - 1))

        a = float(a_vox[x_px])
        b = float(b_vox[x_px])

        ell = (((yy - y0) / a) ** 2 + ((zz - z0) / b) ** 2) <= 1.0
        vol[xv] = circle & ell

    return vol, mm_per_px, voxel_mm


def volume_to_points_mm(volume_bool: np.ndarray, voxel_mm: float, center_yz: bool = True,
                        max_points: int = 500_000):
    pts = np.argwhere(volume_bool)
    if pts.size == 0:
        return None

    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]

    X, Y, Z = volume_bool.shape
    cy = (Y - 1) / 2.0
    cz = (Z - 1) / 2.0

    xs = pts[:, 0].astype(np.float32) * voxel_mm
    ys = pts[:, 1].astype(np.float32)
    zs = pts[:, 2].astype(np.float32)

    if center_yz:
        ys = (ys - cy) * voxel_mm
        zs = (zs - cz) * voxel_mm
    else:
        ys = ys * voxel_mm
        zs = zs * voxel_mm

    return np.c_[xs, ys, zs]


# ==========================================
# PyVista: live animacja + zostaw ostatnią klatkę
# ==========================================
def pv_live_animate_keep_last(frames_data,
                              center_yz: bool = True,
                              max_points: int = 200_000,
                              point_size: float = 3.0,
                              pause_s: float = 0.05,
                              zoom_out: float = 1.8):
    def safe_pts(vol, vox):
        pts = volume_to_points_mm(vol, vox, center_yz=center_yz, max_points=max_points)
        if pts is None:
            return np.empty((0, 3), dtype=np.float32)
        return pts.astype(np.float32, copy=False)

    p = pv.Plotter(shape=(1, 2), window_size=(1500, 720), off_screen=False, title="Bubbles (live)")

    fd0 = frames_data[0]
    poly_a = pv.PolyData(safe_pts(fd0["vol_12"], fd0["vox_12"]))
    poly_b = pv.PolyData(safe_pts(fd0["vol_34"], fd0["vox_34"]))

    p.subplot(0, 0)
    p.add_text("Bubble 3D (tube1&2)", font_size=12)
    p.add_points(poly_a, render_points_as_spheres=True, point_size=point_size)
    p.add_axes()
    p.show_grid()

    p.subplot(0, 1)
    p.add_text("Bubble 3D (tube3&4)", font_size=12)
    p.add_points(poly_b, render_points_as_spheres=True, point_size=point_size)
    p.add_axes()
    p.show_grid()

    p.link_views()

    # --- AUTO CAMERA: oddal widok zanim zacznie się animacja ---
    p.reset_camera()

    # twarde oddalenie (distance)
    try:
        p.camera.distance = float(p.camera.distance) * float(zoom_out)
    except Exception:
        pass

    # dodatkowe "zoom out" (mniej niż 1 oddala)
    try:
        p.camera.zoom(1.0 / float(zoom_out))
    except Exception:
        pass

    try:
        p.show(auto_close=False, interactive_update=True)
    except TypeError:
        # fallback: cannot animate live in this version -> just show first frame blocking
        p.show(auto_close=True)
        return

    for fd in frames_data:
        poly_a.points = safe_pts(fd["vol_12"], fd["vox_12"])
        poly_b.points = safe_pts(fd["vol_34"], fd["vox_34"])

        (poly_a.modified() if hasattr(poly_a, "modified") else poly_a.Modified())
        (poly_b.modified() if hasattr(poly_b, "modified") else poly_b.Modified())

        try:
            p.title = fd["title"]
        except Exception:
            pass

        p.render()
        if hasattr(p, "process_events"):
            p.process_events()
        time.sleep(pause_s)

    # NOW keep last frame until window is closed:
    # This call blocks and keeps the current scene (last frame).
    p.show(auto_close=True)


# ==========================================
# Per-pair reconstruction WITHOUT STICKING
# - connected components in rectified masks
# - match TOP<->SIDE components by X overlap
# - reconstruct each bubble separately, OR volumes
# ==========================================
def reconstruct_pair_no_stick(rect_top: np.ndarray,
                              rect_side: np.ndarray,
                              diameter_mm: float,
                              voxel_mm: float,
                              smooth_sigma_x: float,
                              min_radius_vox: float,
                              min_area_cc: int = 80,
                              iou_thr: float = 0.15):
    top_comps = connected_components(rect_top, min_area=min_area_cc)
    side_comps = connected_components(rect_side, min_area=min_area_cc)

    pairs = match_components_by_x(top_comps, side_comps, iou_thr=iou_thr)

    vol_out = None
    voxel_out = None

    for mt, ms, iou in pairs:
        v, _, vox = build_volume_elliptic_from_two_masks(
            mt, ms,
            diameter_mm=diameter_mm,
            voxel_mm=voxel_mm,
            smooth_sigma_x=smooth_sigma_x,
            min_radius_vox=min_radius_vox
        )
        if vol_out is None:
            vol_out = v
            voxel_out = vox
        else:
            # ensure same shape (should be, because same H,W -> same vol dims)
            if v.shape == vol_out.shape:
                vol_out |= v
            else:
                # rare: shape mismatch due to rounding (very unlikely). pad to max.
                X = max(vol_out.shape[0], v.shape[0])
                Y = max(vol_out.shape[1], v.shape[1])
                Z = max(vol_out.shape[2], v.shape[2])
                vv = np.zeros((X, Y, Z), dtype=bool)
                oo = np.zeros((X, Y, Z), dtype=bool)
                oo[:vol_out.shape[0], :vol_out.shape[1], :vol_out.shape[2]] = vol_out
                vv[:v.shape[0], :v.shape[1], :v.shape[2]] = v
                vol_out = oo | vv
                voxel_out = vox  # close enough

    if vol_out is None:
        # empty result
        vol_out = np.zeros((1, 1, 1), dtype=bool)
        voxel_out = (diameter_mm / float(rect_top.shape[0])) if rect_top.shape[0] > 0 else 1.0

    return vol_out, voxel_out, len(pairs)


# ==========================================
# MAIN: wybór startu po numerze klatki (1-based)
# ==========================================
def main(dataset_dir="bubble.coco/train",
         coco_file="_annotations.coco.json",
         start_frame=1,
         n_frames=40):

    coco = load_coco(os.path.join(dataset_dir, coco_file))
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    bubble_cat_id = None
    for c in categories:
        if c["name"] == "bubble":
            bubble_cat_id = c["id"]
            break
    if bubble_cat_id is None:
        raise RuntimeError("Nie znaleziono kategorii 'bubble'")

    images_sorted = sorted(images, key=lambda d: d.get("file_name", ""))

    start_idx = max(0, int(start_frame) - 1)  # 1-based -> 0-based
    end_idx = min(len(images_sorted), start_idx + max(1, int(n_frames)))
    images_sel = images_sorted[start_idx:end_idx]

    print(f"Start frame (1-based) = {start_frame}  -> idx={start_idx}")
    print(f"Biorę klatki: [{start_idx}:{end_idx}] z {len(images_sorted)} total")
    if not images_sel:
        raise RuntimeError("Zakres start_frame/n_frames poza listą obrazów")

    # tube1=TOP, tube2=SIDE, tube3=TOP, tube4=SIDE
    tubes = [
        {"a_top": 0.007, "b_top": 50,  "a_bot": 0.007, "b_bot": 100},
        {"a_top": 0.019, "b_top": 105, "a_bot": 0.019, "b_bot": 155},
        {"a_top": 0.005, "b_top": 322, "a_bot": 0.005, "b_bot": 380},
        {"a_top": 0.005, "b_top": 408, "a_bot": 0.005, "b_bot": 470},
    ]

    MARGIN_PX = 2.0
    INNER_MARGIN_PX = 2.0
    KEEP_ASPECT = True

    DIAMETER_MM = 20.0
    VOXEL_MM = None  # None => voxel_mm = mm_per_px
    SMOOTH_SIGMA_X = 2.0
    MIN_RADIUS_VOX = 0.8

    # per-bubble settings:
    MIN_AREA_CC = 80
    IOU_THR = 0.15

    frames_data = []

    for local_i, img_info in enumerate(images_sel, start=0):
        global_i = start_idx + local_i + 1  # 1-based do logu
        img_id = img_info["id"]
        img_path = os.path.join(dataset_dir, img_info["file_name"])

        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[WARN] Nie mogę wczytać: {img_path} — pomijam")
            continue

        h, w = gray.shape
        X_LEFT = 0
        X_RIGHT = w - 1

        bubble_anns = [a for a in annotations if a["image_id"] == img_id and a["category_id"] == bubble_cat_id]
        tube_anns = {0: [], 1: [], 2: [], 3: []}

        for ann in bubble_anns:
            bbox = ann.get("bbox", None)
            if bbox is None:
                continue
            x, y, w_box, h_box = map(float, bbox)
            xc = x + 0.5 * w_box
            yc = y + 0.5 * h_box

            for i_tube, tube in enumerate(tubes):
                if point_in_tube(xc, yc, tube, margin_px=MARGIN_PX):
                    tube_anns[i_tube].append(ann)
                    break

        # --- build tube masks (still OK), then rectify pipeline (same as before) ---
        mask1_orig = build_bubble_mask_from_anns(tube_anns[0], h, w)
        mask2_orig = build_bubble_mask_from_anns(tube_anns[1], h, w)
        rect_top_12, rect_side_12, _, _ = rectify_and_align_pair(
            mask1_orig, tubes[0],
            mask2_orig, tubes[1],
            x_left=X_LEFT, x_right=X_RIGHT,
            inner_margin_px=INNER_MARGIN_PX,
            keep_aspect=KEEP_ASPECT
        )

        mask3_orig = build_bubble_mask_from_anns(tube_anns[2], h, w)
        mask4_orig = build_bubble_mask_from_anns(tube_anns[3], h, w)
        rect_top_34, rect_side_34, _, _ = rectify_and_align_pair(
            mask3_orig, tubes[2],
            mask4_orig, tubes[3],
            x_left=X_LEFT, x_right=X_RIGHT,
            inner_margin_px=INNER_MARGIN_PX,
            keep_aspect=KEEP_ASPECT
        )

        # --- per-bubble reconstruction to avoid sticking ---
        vol_12, voxel_mm_12, n_pairs_12 = reconstruct_pair_no_stick(
            rect_top_12, rect_side_12,
            diameter_mm=DIAMETER_MM,
            voxel_mm=VOXEL_MM,
            smooth_sigma_x=SMOOTH_SIGMA_X,
            min_radius_vox=MIN_RADIUS_VOX,
            min_area_cc=MIN_AREA_CC,
            iou_thr=IOU_THR
        )

        vol_34, voxel_mm_34, n_pairs_34 = reconstruct_pair_no_stick(
            rect_top_34, rect_side_34,
            diameter_mm=DIAMETER_MM,
            voxel_mm=VOXEL_MM,
            smooth_sigma_x=SMOOTH_SIGMA_X,
            min_radius_vox=MIN_RADIUS_VOX,
            min_area_cc=MIN_AREA_CC,
            iou_thr=IOU_THR
        )

        print(f"[frame {global_i} ({local_i+1}/{len(images_sel)})] {img_info['file_name']} | "
              f"pairs12={n_pairs_12} filled12={int(vol_12.sum())} | "
              f"pairs34={n_pairs_34} filled34={int(vol_34.sum())}")

        frames_data.append({
            "title": f"Frame {global_i}: {img_info['file_name']}",
            "vol_12": vol_12,
            "vox_12": voxel_mm_12,
            "vol_34": vol_34,
            "vox_34": voxel_mm_34,
        })

    if not frames_data:
        raise RuntimeError("Brak poprawnie przetworzonych klatek (nie wczytano obrazów?)")

    # Live preview: animates + KEEPS LAST FRAME until you close the window
    pv_live_animate_keep_last(
        frames_data,
        center_yz=True,
        max_points=200_000,
        point_size=3.0,
        pause_s=0.25
    )


if __name__ == "__main__":
    # start_frame = numer w posortowanej liście (1-based)
    main(
        start_frame=100,  # <- ustaw np. 475
        n_frames=150
    )