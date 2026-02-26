import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


# =========================
# PyVista helpers
# =========================
def _volume_to_points_mm(volume_bool: np.ndarray, voxel_mm: float, center_yz: bool = True,
                         max_points: int = 500_000):
    pts = np.argwhere(volume_bool)  # (x,y,z) voxel indices
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


def pv_plot_two_pointclouds_side_by_side(vol_a: np.ndarray, vol_b: np.ndarray, voxel_mm: float,
                                        center_yz: bool = True, max_points: int = 500_000,
                                        point_size: float = 3.0,
                                        title_a: str = "Intersection 1-2",
                                        title_b: str = "Intersection 3-4"):
    pts_a = _volume_to_points_mm(vol_a, voxel_mm, center_yz=center_yz, max_points=max_points)
    pts_b = _volume_to_points_mm(vol_b, voxel_mm, center_yz=center_yz, max_points=max_points)

    p = pv.Plotter(shape=(1, 2), window_size=(1400, 700), title="Intersections (side-by-side)")

    # left
    p.subplot(0, 0)
    p.add_text(title_a, font_size=12)
    if pts_a is None:
        p.add_text("EMPTY", font_size=18, position="upper_left")
    else:
        cloud_a = pv.PolyData(pts_a)
        p.add_points(cloud_a, render_points_as_spheres=True, point_size=point_size)
    p.add_axes()
    p.show_grid()

    # right
    p.subplot(0, 1)
    p.add_text(title_b, font_size=12)
    if pts_b is None:
        p.add_text("EMPTY", font_size=18, position="upper_left")
    else:
        cloud_b = pv.PolyData(pts_b)
        p.add_points(cloud_b, render_points_as_spheres=True, point_size=point_size)
    p.add_axes()
    p.show_grid()

    p.link_views()
    p.show()


def pv_plot_isosurface(volume_bool: np.ndarray, voxel_mm: float, center_yz: bool = True,
                       smooth_iters: int = 0, title="Isosurface"):
    if volume_bool.ndim != 3:
        raise ValueError("volume_bool musi być 3D")
    if volume_bool.sum() == 0:
        print("[PyVista] volume pusty.")
        return

    X, Y, Z = volume_bool.shape

    grid = pv.ImageData()
    grid.dimensions = (X + 1, Y + 1, Z + 1)  # points
    grid.spacing = (voxel_mm, voxel_mm, voxel_mm)

    if center_yz:
        cy = (Y - 1) / 2.0
        cz = (Z - 1) / 2.0
        grid.origin = (0.0, -cy * voxel_mm, -cz * voxel_mm)
    else:
        grid.origin = (0.0, 0.0, 0.0)

    # cell occupancy
    occ_cell = volume_bool.astype(np.uint8).ravel(order="F")  # X*Y*Z
    grid.cell_data["occ_cell"] = occ_cell

    # cell -> point (required for contour)
    grid_pt = grid.cell_data_to_point_data(pass_cell_data=False)
    if "occ_cell" not in grid_pt.point_data:
        raise RuntimeError("Nie znalazłem occ_cell w point_data po konwersji.")

    surf = grid_pt.contour(isosurfaces=[0.5], scalars="occ_cell")

    if smooth_iters > 0 and surf.n_points > 0:
        surf = surf.smooth(n_iter=smooth_iters)

    p = pv.Plotter(title=title, window_size=(900, 800))
    p.add_text(title, font_size=12)
    p.add_mesh(surf, opacity=1.0)
    p.add_axes()
    p.show_grid()
    p.show()


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

    # --- POLYGON ---
    if isinstance(seg, list) and len(seg) > 0:
        for poly in seg:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    # --- BBOX fallback ---
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

    # upewnij się, że top jest nad bottom
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
        [0,        0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0,        out_h - 1],
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


# ==========================================
# 3D: projekcje prostopadłe + przecięcie
# ==========================================
def _compute_grid_from_rect(H, W, diameter_mm, voxel_mm=None):
    mm_per_px = diameter_mm / float(H)
    if voxel_mm is None:
        voxel_mm = mm_per_px

    R_vox = int(round((diameter_mm / 2.0) / voxel_mm))
    YZ = 2 * R_vox + 1

    length_mm = W * mm_per_px
    X = max(1, int(round(length_mm / voxel_mm)))

    x_map = np.linspace(0, X - 1, W).round().astype(int)
    cy = R_vox
    cz = R_vox
    return X, YZ, cy, cz, x_map, mm_per_px, voxel_mm


def project_mask_side_view_to_cylinder_volume(mask_side: np.ndarray, diameter_mm: float,
                                             voxel_mm: float = None,
                                             fill_full_chord: bool = True):
    if mask_side.ndim != 2:
        raise ValueError("mask_side musi być 2D.")
    H, W = mask_side.shape

    X, YZ, cy, cz, x_map, mm_per_px, voxel_mm = _compute_grid_from_rect(H, W, diameter_mm, voxel_mm)
    vol = np.zeros((X, YZ, YZ), dtype=bool)

    ys, xs = np.where(mask_side > 0)
    if ys.size == 0:
        return vol, mm_per_px, voxel_mm

    y_center_px = (H - 1) / 2.0
    y_mm = (ys - y_center_px) * mm_per_px
    y_vox = np.round(y_mm / voxel_mm).astype(int) + cy
    y_vox = np.clip(y_vox, 0, YZ - 1)

    R2 = (YZ // 2) ** 2

    for i in range(xs.size):
        x_vox = x_map[xs[i]]
        yv = int(y_vox[i])

        dy = yv - cy
        inside = R2 - dy * dy
        if inside <= 0:
            continue
        z_extent = int(np.floor(np.sqrt(inside)))

        if fill_full_chord:
            z0 = cz - z_extent
            z1 = cz + z_extent
        else:
            z0 = cz
            z1 = cz + z_extent

        z0 = max(0, z0)
        z1 = min(YZ - 1, z1)
        vol[x_vox, yv, z0:z1 + 1] = True

    return vol, mm_per_px, voxel_mm


def project_mask_top_view_to_cylinder_volume(mask_top: np.ndarray, diameter_mm: float,
                                            voxel_mm: float = None,
                                            fill_full_chord: bool = True):
    if mask_top.ndim != 2:
        raise ValueError("mask_top musi być 2D.")
    H, W = mask_top.shape

    X, YZ, cy, cz, x_map, mm_per_px, voxel_mm = _compute_grid_from_rect(H, W, diameter_mm, voxel_mm)
    vol = np.zeros((X, YZ, YZ), dtype=bool)

    zs, xs = np.where(mask_top > 0)
    if zs.size == 0:
        return vol, mm_per_px, voxel_mm

    z_center_px = (H - 1) / 2.0
    z_mm = (zs - z_center_px) * mm_per_px
    z_vox = np.round(z_mm / voxel_mm).astype(int) + cz
    z_vox = np.clip(z_vox, 0, YZ - 1)

    R2 = (YZ // 2) ** 2

    for i in range(xs.size):
        x_vox = x_map[xs[i]]
        zv = int(z_vox[i])

        dz = zv - cz
        inside = R2 - dz * dz
        if inside <= 0:
            continue
        y_extent = int(np.floor(np.sqrt(inside)))

        if fill_full_chord:
            y0 = cy - y_extent
            y1 = cy + y_extent
        else:
            y0 = cy
            y1 = cy + y_extent

        y0 = max(0, y0)
        y1 = min(YZ - 1, y1)
        vol[x_vox, y0:y1 + 1, zv] = True

    return vol, mm_per_px, voxel_mm


def _rectify_and_align_pair(mask_a_orig: np.ndarray, tube_a: dict,
                            mask_b_orig: np.ndarray, tube_b: dict,
                            x_left: int, x_right: int, inner_margin_px: float,
                            keep_aspect: bool):
    rect_a, quad_a = crop_and_rectify_tube(
        mask_a_orig, tube_a, x_left=x_left, x_right=x_right,
        inner_margin_px=inner_margin_px, interp=cv2.INTER_NEAREST
    )
    rect_b, quad_b = crop_and_rectify_tube(
        mask_b_orig, tube_b, x_left=x_left, x_right=x_right,
        inner_margin_px=inner_margin_px, interp=cv2.INTER_NEAREST
    )

    H_target = max(rect_a.shape[0], rect_b.shape[0])
    rect_a = resize_to_match_height(rect_a, H_target, keep_aspect=keep_aspect, is_mask=True)
    rect_b = resize_to_match_height(rect_b, H_target, keep_aspect=keep_aspect, is_mask=True)

    W_target = min(rect_a.shape[1], rect_b.shape[1])
    rect_a = center_crop_or_pad_width(rect_a, W_target, pad_value=0)
    rect_b = center_crop_or_pad_width(rect_b, W_target, pad_value=0)

    return rect_a, rect_b, quad_a, quad_b


def _unify_shapes(vol_a: np.ndarray, vol_b: np.ndarray):
    if vol_a.shape == vol_b.shape:
        return vol_a, vol_b
    Xc = min(vol_a.shape[0], vol_b.shape[0])
    Yc = min(vol_a.shape[1], vol_b.shape[1])
    Zc = min(vol_a.shape[2], vol_b.shape[2])
    return vol_a[:Xc, :Yc, :Zc], vol_b[:Xc, :Yc, :Zc]


# ==========================================
# EXPORT: szybki PLY (point cloud z voxeli)
# ==========================================
def export_volume_to_ply_pointcloud(volume_bool: np.ndarray,
                                    out_path: str,
                                    voxel_mm: float,
                                    center_yz: bool = True,
                                    max_points: int = 1_000_000):
    pts = np.argwhere(volume_bool)
    n = pts.shape[0]
    if n == 0:
        print("[PLY] volume pusty, nic nie zapisuję.")
        return

    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        pts = pts[idx]
        n = pts.shape[0]
        print(f"[PLY] Ograniczam liczbę punktów do {n} (losowanie).")

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

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(f"{xs[i]:.6f} {ys[i]:.6f} {zs[i]:.6f}\n")

    print(f"[PLY] Zapisano: {out_path} (punkty: {n})")


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

    # 4 RURKI: (1,2) para pierwsza, (3,4) para druga
    tubes = [
        {"a_top": 0.007, "b_top": 50,  "a_bot": 0.007, "b_bot": 100},   # tube 1 (TOP VIEW)
        {"a_top": 0.019, "b_top": 105, "a_bot": 0.019, "b_bot": 155},   # tube 2 (SIDE VIEW)
        {"a_top": 0.005, "b_top": 322, "a_bot": 0.005, "b_bot": 380},   # tube 3 (TOP VIEW)
        {"a_top": 0.005, "b_top": 408, "a_bot": 0.005, "b_bot": 470},   # tube 4 (SIDE VIEW)
    ]

    MARGIN_PX = 2.0
    INNER_MARGIN_PX = 2.0
    X_LEFT = 0
    X_RIGHT = w - 1
    KEEP_ASPECT = True

    # ===== 3D cylinder =====
    DIAMETER_MM = 20.0
    VOXEL_MM = None
    FILL_FULL_CHORD = True

    # ===== STEP 1: pokaż rurki =====
    vis1 = draw_tubes(rgb, tubes)
    show_step(vis1, "KROK 1: 4 rurki (linie ax+b)")

    # ===== STEP 2: przypisz bąble do rurek =====
    bubble_anns = [
        a for a in annotations
        if a["image_id"] == img_id and a["category_id"] == bubble_cat_id
    ]

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
    colors = [(255, 0, 0), (0, 255, 0),
              (0, 0, 255), (255, 255, 0)]
    for tube_id in range(4):
        for ann in tube_anns[tube_id]:
            m = ann_to_mask(ann, h, w)
            vis2 = overlay_mask(vis2, m, color=colors[tube_id], alpha=0.45)

    show_step(vis2, "KROK 2: Bąble przypisane do rurek")
    print("Liczba bąbli w rurkach:", [len(tube_anns[i]) for i in range(4)])

    # ============================
    # PARA 1: tube1 + tube2
    # ============================
    mask1_orig = build_bubble_mask_from_anns(tube_anns[0], h, w)  # TOP
    mask2_orig = build_bubble_mask_from_anns(tube_anns[1], h, w)  # SIDE

    rect1_s, rect2_s, quad1, quad2 = _rectify_and_align_pair(
        mask1_orig, tubes[0],
        mask2_orig, tubes[1],
        x_left=X_LEFT, x_right=X_RIGHT,
        inner_margin_px=INNER_MARGIN_PX,
        keep_aspect=KEEP_ASPECT
    )

    show_step(rect1_s, "PAIR 1: TOP (tube1) po pipeline", cmap="gray")
    show_step(rect2_s, "PAIR 1: SIDE (tube2) po pipeline", cmap="gray")

    vol_top_12, mm_per_px_12, voxel_mm_12 = project_mask_top_view_to_cylinder_volume(
        rect1_s, diameter_mm=DIAMETER_MM, voxel_mm=VOXEL_MM, fill_full_chord=FILL_FULL_CHORD
    )
    vol_side_12, _, _ = project_mask_side_view_to_cylinder_volume(
        rect2_s, diameter_mm=DIAMETER_MM, voxel_mm=voxel_mm_12, fill_full_chord=FILL_FULL_CHORD
    )
    vol_top_12, vol_side_12 = _unify_shapes(vol_top_12, vol_side_12)
    inter_12 = np.logical_and(vol_top_12, vol_side_12)

    print("\n[PAIR 1-2]")
    print(f"[3D] voxel_mm={voxel_mm_12:.6f} mm/voxel  |  mm_per_px={mm_per_px_12:.6f} mm/px")
    print(f"[3D] vol_top_12 filled = {int(vol_top_12.sum())}")
    print(f"[3D] vol_side_12 filled = {int(vol_side_12.sum())}")
    print(f"[3D] inter_12 filled = {int(inter_12.sum())}")

    # ============================
    # PARA 2: tube3 + tube4
    # ============================
    mask3_orig = build_bubble_mask_from_anns(tube_anns[2], h, w)  # TOP
    mask4_orig = build_bubble_mask_from_anns(tube_anns[3], h, w)  # SIDE

    rect3_s, rect4_s, quad3, quad4 = _rectify_and_align_pair(
        mask3_orig, tubes[2],
        mask4_orig, tubes[3],
        x_left=X_LEFT, x_right=X_RIGHT,
        inner_margin_px=INNER_MARGIN_PX,
        keep_aspect=KEEP_ASPECT
    )

    show_step(rect3_s, "PAIR 3: TOP (tube3) po pipeline", cmap="gray")
    show_step(rect4_s, "PAIR 4: SIDE (tube4) po pipeline", cmap="gray")

    # Uwaga: to jest inna para, ale nadal walec D=20 mm. voxel_mm bierzemy z tej pary (albo narzuć z 12).
    vol_top_34, mm_per_px_34, voxel_mm_34 = project_mask_top_view_to_cylinder_volume(
        rect3_s, diameter_mm=DIAMETER_MM, voxel_mm=VOXEL_MM, fill_full_chord=FILL_FULL_CHORD
    )
    vol_side_34, _, _ = project_mask_side_view_to_cylinder_volume(
        rect4_s, diameter_mm=DIAMETER_MM, voxel_mm=voxel_mm_34, fill_full_chord=FILL_FULL_CHORD
    )
    vol_top_34, vol_side_34 = _unify_shapes(vol_top_34, vol_side_34)
    inter_34 = np.logical_and(vol_top_34, vol_side_34)

    print("\n[PAIR 3-4]")
    print(f"[3D] voxel_mm={voxel_mm_34:.6f} mm/voxel  |  mm_per_px={mm_per_px_34:.6f} mm/px")
    print(f"[3D] vol_top_34 filled = {int(vol_top_34.sum())}")
    print(f"[3D] vol_side_34 filled = {int(vol_side_34.sum())}")
    print(f"[3D] inter_34 filled = {int(inter_34.sum())}")

    # ============================
    # PyVista: pokaż obok siebie dwie chmury (intersection 1-2 oraz 3-4)
    # ============================
    # Jeśli chcesz, żeby obie chmury miały IDENTYCZNĄ skalę voxeli, możesz narzucić voxel_mm=voxel_mm_12
    # i przeliczać drugą parę na to samo voxel_mm. Na razie pokazuję "jak jest" dla każdej pary.
    # Żeby wyświetlić obok siebie w jednym oknie, trzeba mieć jeden voxel_mm (jednostki sceny).
    # Najprościej: użyć voxel_mm_12 i voxel_mm_34 osobno -> ale wtedy jedna scena nie ma sensu.
    # Rozwiązanie: wyświetlamy obok siebie, ale w tej samej jednostce (mm) — więc to OK,
    # bo oba są w mm. W samej funkcji konwertujemy do mm i tyle.
    pv_plot_two_pointclouds_side_by_side(
        inter_12, inter_34,
        voxel_mm=1.0,  # UWAGA: tu voxel_mm nie jest używany jako spacing, bo punkty są już w mm poniżej.
        center_yz=True,
        max_points=500_000,
        point_size=3.0,
        title_a="Intersection (tube1 & tube2)",
        title_b="Intersection (tube3 & tube4)"
    )

    # Poprawka: w powyższej funkcji _volume_to_points_mm używa voxel_mm do przeliczenia.
    # Musimy więc wywołać ją z właściwym voxel_mm per volume.
    # Dlatego robimy dedykowane okno z dwoma subplotami manualnie:

    # --- manual side-by-side with correct voxel sizes ---
    pts12 = _volume_to_points_mm(inter_12, voxel_mm_12, center_yz=True, max_points=500_000)
    pts34 = _volume_to_points_mm(inter_34, voxel_mm_34, center_yz=True, max_points=500_000)

    p = pv.Plotter(shape=(1, 2), window_size=(1400, 700), title="Intersections 1-2 and 3-4")
    p.subplot(0, 0)
    p.add_text("Intersection (tube1 & tube2)", font_size=12)
    if pts12 is not None:
        p.add_points(pv.PolyData(pts12), render_points_as_spheres=True, point_size=3.0)
    else:
        p.add_text("EMPTY", font_size=18, position="upper_left")
    p.add_axes()
    p.show_grid()

    p.subplot(0, 1)
    p.add_text("Intersection (tube3 & tube4)", font_size=12)
    if pts34 is not None:
        p.add_points(pv.PolyData(pts34), render_points_as_spheres=True, point_size=3.0)
    else:
        p.add_text("EMPTY", font_size=18, position="upper_left")
    p.add_axes()
    p.show_grid()

    p.link_views()
    p.show()

    # (opcjonalnie) isosurface osobno:
    # pv_plot_isosurface(inter_12, voxel_mm=voxel_mm_12, center_yz=True, smooth_iters=20, title="Isosurface 1-2")
    # pv_plot_isosurface(inter_34, voxel_mm=voxel_mm_34, center_yz=True, smooth_iters=20, title="Isosurface 3-4")

    # (opcjonalnie) eksport PLY:
    export_volume_to_ply_pointcloud(inter_12, "bubble_intersection_12.ply", voxel_mm=voxel_mm_12, center_yz=True)
    export_volume_to_ply_pointcloud(inter_34, "bubble_intersection_34.ply", voxel_mm=voxel_mm_34, center_yz=True)


if __name__ == "__main__":
    main()