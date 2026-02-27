import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

# (opcjonalnie) SciPy do wygładzania profili wzdłuż X
try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ==========================================
# STABILNA WIZUALIZACJA (bez znikania okien)
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


# ============================================================
# NEW: build 3D volume from TWO silhouettes using ELLIPTIC cross-sections
# ============================================================
def _profile_from_mask_columns(mask_2d: np.ndarray):
    """
    mask_2d: uint8 [H,W]
    returns:
      low[W], high[W], center[W], half[W], valid[W]
    where vertical axis is rows (0..H-1)
    """
    H, W = mask_2d.shape
    low = np.zeros(W, dtype=np.int32)
    high = np.zeros(W, dtype=np.int32)
    center = np.zeros(W, dtype=np.float32)
    half = np.zeros(W, dtype=np.float32)
    valid = np.zeros(W, dtype=bool)

    for x in range(W):
        ys = np.where(mask_2d[:, x] > 0)[0]
        if ys.size == 0:
            valid[x] = False
            low[x] = 0
            high[x] = -1
            center[x] = (H - 1) / 2.0
            half[x] = 0.0
            continue
        y0 = int(ys.min())
        y1 = int(ys.max())
        low[x] = y0
        high[x] = y1
        center[x] = 0.5 * (y0 + y1)
        half[x] = 0.5 * (y1 - y0)
        valid[x] = True

    return low, high, center, half, valid


def _fill_missing_1d(arr: np.ndarray, valid: np.ndarray):
    """Linear interpolate missing values in 1D."""
    x = np.arange(arr.size)
    if valid.sum() == 0:
        return arr.copy()
    out = arr.astype(np.float32).copy()
    out[~valid] = np.interp(x[~valid], x[valid], out[valid])
    return out


def build_volume_elliptic_from_two_masks(mask_top_xz: np.ndarray,
                                        mask_side_xy: np.ndarray,
                                        diameter_mm: float,
                                        voxel_mm: float = None,
                                        smooth_sigma_x: float = 2.0,
                                        min_radius_vox: float = 0.8):
    """
    mask_top_xz:  (H,W) where vertical axis is Z, horizontal is X
    mask_side_xy: (H,W) where vertical axis is Y, horizontal is X
    returns: volume_bool (X, YZ, YZ), mm_per_px, voxel_mm
    """
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

    # Profiles from masks:
    # TOP gives Z extents (rows are Z)
    _, _, zc_px, zhalf_px, zvalid = _profile_from_mask_columns(mask_top_xz)
    # SIDE gives Y extents (rows are Y)
    _, _, yc_px, yhalf_px, yvalid = _profile_from_mask_columns(mask_side_xy)

    valid_both = zvalid & yvalid
    # interpolate missing columns (keeps continuity)
    zc_px = _fill_missing_1d(zc_px, valid_both)
    yc_px = _fill_missing_1d(yc_px, valid_both)
    zhalf_px = _fill_missing_1d(zhalf_px, valid_both)
    yhalf_px = _fill_missing_1d(yhalf_px, valid_both)

    # convert to vox radii and centers (in voxel coords)
    px_center = (H - 1) / 2.0
    z0_mm = (zc_px - px_center) * mm_per_px
    y0_mm = (yc_px - px_center) * mm_per_px

    z0_vox = np.round(z0_mm / voxel_mm).astype(np.int32) + cz
    y0_vox = np.round(y0_mm / voxel_mm).astype(np.int32) + cy

    b_vox = np.maximum(min_radius_vox, (zhalf_px * mm_per_px) / voxel_mm).astype(np.float32)  # Z semi-axis
    a_vox = np.maximum(min_radius_vox, (yhalf_px * mm_per_px) / voxel_mm).astype(np.float32)  # Y semi-axis

    # smooth along X to avoid jaggedness
    if smooth_sigma_x and smooth_sigma_x > 0:
        if HAS_SCIPY:
            a_vox = gaussian_filter1d(a_vox, sigma=smooth_sigma_x).astype(np.float32)
            b_vox = gaussian_filter1d(b_vox, sigma=smooth_sigma_x).astype(np.float32)
            y0_vox = np.round(gaussian_filter1d(y0_vox.astype(np.float32), sigma=smooth_sigma_x)).astype(np.int32)
            z0_vox = np.round(gaussian_filter1d(z0_vox.astype(np.float32), sigma=smooth_sigma_x)).astype(np.int32)
        else:
            # simple moving average fallback
            k = int(max(3, 2 * round(smooth_sigma_x) + 1))
            k = k if (k % 2 == 1) else k + 1
            ker = np.ones(k, dtype=np.float32) / k
            a_vox = np.convolve(a_vox, ker, mode="same").astype(np.float32)
            b_vox = np.convolve(b_vox, ker, mode="same").astype(np.float32)

    vol = np.zeros((X, YZ, YZ), dtype=bool)

    # precompute circle mask of the pipe cross-section
    yy, zz = np.meshgrid(np.arange(YZ), np.arange(YZ), indexing="ij")
    circle = ((yy - cy) ** 2 + (zz - cz) ** 2) <= (R_vox ** 2)

    for x_px in range(W):
        xv = int(x_map[x_px])
        # clamp centers
        y0 = int(np.clip(y0_vox[x_px], 0, YZ - 1))
        z0 = int(np.clip(z0_vox[x_px], 0, YZ - 1))

        a = float(a_vox[x_px])
        b = float(b_vox[x_px])

        # if either radius is tiny -> skip
        if a < 0.5 or b < 0.5:
            continue

        # ellipse mask in YZ
        ell = (((yy - y0) / a) ** 2 + ((zz - z0) / b) ** 2) <= 1.0
        vol[xv] = circle & ell

    return vol, mm_per_px, voxel_mm


# ==========================================
# PyVista: two pointclouds side-by-side
# ==========================================
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


def pv_show_two_pointclouds(vol_a: np.ndarray, voxel_mm_a: float,
                            vol_b: np.ndarray, voxel_mm_b: float,
                            title_a: str, title_b: str,
                            center_yz: bool = True,
                            max_points: int = 500_000,
                            point_size: float = 3.0):
    pts_a = volume_to_points_mm(vol_a, voxel_mm_a, center_yz=center_yz, max_points=max_points)
    pts_b = volume_to_points_mm(vol_b, voxel_mm_b, center_yz=center_yz, max_points=max_points)

    p = pv.Plotter(shape=(1, 2), window_size=(1500, 720), title="Bubbles (elliptic reconstruction)")

    p.subplot(0, 0)
    p.add_text(title_a, font_size=12)
    if pts_a is None:
        p.add_text("EMPTY", font_size=18, position="upper_left")
    else:
        p.add_points(pv.PolyData(pts_a), render_points_as_spheres=True, point_size=point_size)
    p.add_axes()
    p.show_grid()

    p.subplot(0, 1)
    p.add_text(title_b, font_size=12)
    if pts_b is None:
        p.add_text("EMPTY", font_size=18, position="upper_left")
    else:
        p.add_points(pv.PolyData(pts_b), render_points_as_spheres=True, point_size=point_size)
    p.add_axes()
    p.show_grid()

    p.link_views()
    p.show()


# ==========================================
# MAIN
# ==========================================
def main(dataset_dir="bubble.coco/train",
         coco_file="_annotations.coco.json",
         image_number=60):

    coco = load_coco(os.path.join(dataset_dir, coco_file))
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    bubble_cat_id = None
    for c in categories:
        if c["name"] == "bubble":
            bubble_cat_id = c["id"]
    if bubble_cat_id is None:
        raise RuntimeError("Nie znaleziono kategorii 'bubble'")

    idx = image_number - 1
    if idx < 0 or idx >= len(images):
        raise RuntimeError("Nie ma tylu zdjęć w dataset")

    img_info = images[idx]
    img_id = img_info["id"]
    img_path = os.path.join(dataset_dir, img_info["file_name"])

    print(f"Używam zdjęcia #{image_number}: {img_info['file_name']}")

    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError("Nie mogę wczytać obrazu")

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    h, w = gray.shape

    # tube1=TOP, tube2=SIDE, tube3=TOP, tube4=SIDE
    tubes = [
        {"a_top": 0.007, "b_top": 50,  "a_bot": 0.007, "b_bot": 100},
        {"a_top": 0.019, "b_top": 105, "a_bot": 0.019, "b_bot": 155},
        {"a_top": 0.005, "b_top": 322, "a_bot": 0.005, "b_bot": 380},
        {"a_top": 0.005, "b_top": 408, "a_bot": 0.005, "b_bot": 470},
    ]

    MARGIN_PX = 2.0
    INNER_MARGIN_PX = 2.0
    X_LEFT = 0
    X_RIGHT = w - 1
    KEEP_ASPECT = True

    DIAMETER_MM = 20.0
    VOXEL_MM = None

    # reconstruction params (tune)
    SMOOTH_SIGMA_X = 2.0   # wygładzenie a(x), b(x), y0(x), z0(x)
    MIN_RADIUS_VOX = 0.8   # nie pozwól, żeby elipsa zapadła się do zera

    vis1 = draw_tubes(rgb, tubes)
    show_step(vis1, "KROK 1: 4 rurki (linie ax+b)")

    bubble_anns = [a for a in annotations if a["image_id"] == img_id and a["category_id"] == bubble_cat_id]
    tube_anns = {0: [], 1: [], 2: [], 3: []}

    for ann in bubble_anns:
        bbox = ann.get("bbox", None)
        if bbox is None:
            continue
        x, y, w_box, h_box = map(float, bbox)
        xc = x + 0.5 * w_box
        yc = y + 0.5 * h_box

        for i, tube in enumerate(tubes):
            if point_in_tube(xc, yc, tube, margin_px=MARGIN_PX):
                tube_anns[i].append(ann)
                break

    vis2 = vis1.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for tube_id in range(4):
        for ann in tube_anns[tube_id]:
            m = ann_to_mask(ann, h, w)
            vis2 = overlay_mask(vis2, m, color=colors[tube_id], alpha=0.45)
    show_step(vis2, "KROK 2: Bąble przypisane do rurek")
    print("Liczba bąbli w rurkach:", [len(tube_anns[i]) for i in range(4)])

    # ============================
    # PAIR 1: (tube1 TOP) + (tube2 SIDE)
    # ============================
    mask1_orig = build_bubble_mask_from_anns(tube_anns[0], h, w)
    mask2_orig = build_bubble_mask_from_anns(tube_anns[1], h, w)

    rect_top_12, rect_side_12, _, _ = rectify_and_align_pair(
        mask1_orig, tubes[0],
        mask2_orig, tubes[1],
        x_left=X_LEFT, x_right=X_RIGHT,
        inner_margin_px=INNER_MARGIN_PX,
        keep_aspect=KEEP_ASPECT
    )
    show_step(rect_top_12, "PAIR 1-2: TOP po pipeline", cmap="gray")
    show_step(rect_side_12, "PAIR 1-2: SIDE po pipeline", cmap="gray")

    vol_12, mm_per_px_12, voxel_mm_12 = build_volume_elliptic_from_two_masks(
        rect_top_12, rect_side_12,
        diameter_mm=DIAMETER_MM,
        voxel_mm=VOXEL_MM,
        smooth_sigma_x=SMOOTH_SIGMA_X,
        min_radius_vox=MIN_RADIUS_VOX
    )

    print("\n[PAIR 1-2 elliptic]")
    print(f"mm_per_px={mm_per_px_12:.6f} | voxel_mm={voxel_mm_12:.6f} | filled={int(vol_12.sum())}")

    # ============================
    # PAIR 2: (tube3 TOP) + (tube4 SIDE)
    # ============================
    mask3_orig = build_bubble_mask_from_anns(tube_anns[2], h, w)
    mask4_orig = build_bubble_mask_from_anns(tube_anns[3], h, w)

    rect_top_34, rect_side_34, _, _ = rectify_and_align_pair(
        mask3_orig, tubes[2],
        mask4_orig, tubes[3],
        x_left=X_LEFT, x_right=X_RIGHT,
        inner_margin_px=INNER_MARGIN_PX,
        keep_aspect=KEEP_ASPECT
    )
    show_step(rect_top_34, "PAIR 3-4: TOP po pipeline", cmap="gray")
    show_step(rect_side_34, "PAIR 3-4: SIDE po pipeline", cmap="gray")

    vol_34, mm_per_px_34, voxel_mm_34 = build_volume_elliptic_from_two_masks(
        rect_top_34, rect_side_34,
        diameter_mm=DIAMETER_MM,
        voxel_mm=VOXEL_MM,
        smooth_sigma_x=SMOOTH_SIGMA_X,
        min_radius_vox=MIN_RADIUS_VOX
    )

    print("\n[PAIR 3-4 elliptic]")
    print(f"mm_per_px={mm_per_px_34:.6f} | voxel_mm={voxel_mm_34:.6f} | filled={int(vol_34.sum())}")

    # ============================
    # PyVista: show side-by-side
    # ============================
    pv_show_two_pointclouds(
        vol_12, voxel_mm_12,
        vol_34, voxel_mm_34,
        title_a="Bubble 3D (tube1&2) - elliptic",
        title_b="Bubble 3D (tube3&4) - elliptic",
        center_yz=True,
        max_points=500_000,
        point_size=3.0
    )


if __name__ == "__main__":
    main()