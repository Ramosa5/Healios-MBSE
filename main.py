import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def alpha_blend_rgb(img1_rgb, img2_rgb, alpha=0.5):
    img1f = img1_rgb.astype(np.float32)
    img2f = img2_rgb.astype(np.float32)
    out = img1f * alpha + img2f * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


# ==========================================
# 3D: projekcje prostopadłe + przecięcie
# Ustalamy osie:
# - X: oś rurki (wzdłuż szerokości obrazu W)
# - Y,Z: przekrój kołowy
#
# Widok Z GÓRY (tube1): obraz to X vs Z, a my wypełniamy chord w osi Y
# Widok Z BOKU (tube2): obraz to X vs Y, a my wypełniamy chord w osi Z
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
    """
    Widok z boku: obraz (X,Y). Dla każdego (x,y) w masce wypełniamy chord w Z.
    """
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

    for i in range(xs.size):
        x_vox = x_map[xs[i]]
        yv = int(y_vox[i])

        dy = yv - cy
        inside = (YZ // 2) ** 2 - dy * dy
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
    """
    Widok z góry: obraz (X,Z). Dla każdego (x,z) w masce wypełniamy chord w Y.
    """
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

    for i in range(xs.size):
        x_vox = x_map[xs[i]]
        zv = int(z_vox[i])

        dz = zv - cz
        inside = (YZ // 2) ** 2 - dz * dz
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


def show_3d_volume_scatter(volume_bool: np.ndarray, title: str, max_points: int = 120000):
    pts = np.argwhere(volume_bool)
    if pts.shape[0] == 0:
        print("Brak punktów w objętości (volume jest puste).")
        return

    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=1)
    ax.set_title(title)
    ax.set_xlabel("X (oś rurki)")
    ax.set_ylabel("Y (przekrój)")
    ax.set_zlabel("Z (przekrój)")
    plt.tight_layout()

    plt.show(block=False)
    plt.pause(0.001)
    plt.waitforbuttonpress()
    plt.close(fig)


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

    # ==========================================
    # 4 RURKI: tube1 = widok z góry, tube2 = widok z boku (tej samej fizycznej rurki)
    # ==========================================
    tubes = [
        {"a_top": 0.007, "b_top": 50,  "a_bot": 0.007, "b_bot": 100},   # tube 1 (TOP VIEW)
        {"a_top": 0.019, "b_top": 105, "a_bot": 0.019, "b_bot": 155},   # tube 2 (SIDE VIEW)
        {"a_top": 0.005, "b_top": 322, "a_bot": 0.005, "b_bot": 380},
        {"a_top": 0.005, "b_top": 408, "a_bot": 0.005, "b_bot": 470},
    ]

    # przypisywanie bąbli do rurek
    MARGIN_PX = 2.0
    # wycinanie "do środka"
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

    # ============================================================
    # KROK 3: ZRÓB DWIE MASKI (tube1 i tube2) i przepuść przez TEN SAM pipeline geometrii
    #         (wytnij po liniach -> wyprostuj -> skaluj do wyższej -> dopasuj szerokość)
    # ============================================================
    mask1_orig = build_bubble_mask_from_anns(tube_anns[0], h, w)  # tube1 (top view)
    mask2_orig = build_bubble_mask_from_anns(tube_anns[1], h, w)  # tube2 (side view)

    view1 = overlay_mask(rgb, mask1_orig, color=(255, 0, 255), alpha=0.55)
    view2 = overlay_mask(rgb, mask2_orig, color=(0, 255, 255), alpha=0.55)

    show_step(view1, "KROK 3A: tube1 (TOP) maska bąbli na oryginale")
    show_step(view2, "KROK 3B: tube2 (SIDE) maska bąbli na oryginale")

    # Wytnij+wyprostuj maski (NEAREST!)
    rect1_mask, quad1 = crop_and_rectify_tube(
        mask1_orig, tubes[0], x_left=X_LEFT, x_right=X_RIGHT,
        inner_margin_px=INNER_MARGIN_PX, interp=cv2.INTER_NEAREST
    )
    rect2_mask, quad2 = crop_and_rectify_tube(
        mask2_orig, tubes[1], x_left=X_LEFT, x_right=X_RIGHT,
        inner_margin_px=INNER_MARGIN_PX, interp=cv2.INTER_NEAREST
    )

    # Skalowanie do wyższej (większej H)
    H_target = max(rect1_mask.shape[0], rect2_mask.shape[0])
    rect1_s = resize_to_match_height(rect1_mask, H_target, keep_aspect=KEEP_ASPECT, is_mask=True)
    rect2_s = resize_to_match_height(rect2_mask, H_target, keep_aspect=KEEP_ASPECT, is_mask=True)

    # Dopasuj szerokość do wspólnej (min) bez dodatkowego rozciągania
    W_target = min(rect1_s.shape[1], rect2_s.shape[1])
    rect1_s = center_crop_or_pad_width(rect1_s, W_target, pad_value=0)
    rect2_s = center_crop_or_pad_width(rect2_s, W_target, pad_value=0)

    show_step(rect1_s, "KROK 3C: maska TOP po pipeline (wycięcie+wyprost+skalowanie)", cmap="gray")
    show_step(rect2_s, "KROK 3D: maska SIDE po pipeline (wycięcie+wyprost+skalowanie)", cmap="gray")

    # Debug: równoległoboki na oryginale
    dbg = rgb.copy()
    cv2.polylines(dbg, [quad1.astype(np.int32)], True, (255, 0, 255), 2)
    cv2.polylines(dbg, [quad2.astype(np.int32)], True, (0, 255, 255), 2)
    show_step(dbg, "DEBUG: Równoległoboki wycinania (magenta=TOP, cyan=SIDE)")

    # ============================================================
    # KROK 4: DWA RZUTY NA WALec (PROSTOPADŁE) + CZĘŚĆ WSPÓLNA
    # ============================================================
    vol_top, mm_per_px_top, voxel_mm_top = project_mask_top_view_to_cylinder_volume(
        rect1_s, diameter_mm=DIAMETER_MM, voxel_mm=VOXEL_MM, fill_full_chord=FILL_FULL_CHORD
    )
    vol_side, mm_per_px_side, voxel_mm_side = project_mask_side_view_to_cylinder_volume(
        rect2_s, diameter_mm=DIAMETER_MM, voxel_mm=voxel_mm_top, fill_full_chord=FILL_FULL_CHORD
    )

    # Grid powinien wyjść taki sam (bo ten sam voxel_mm i ten sam W,H po pipeline)
    if vol_top.shape != vol_side.shape:
        # jeśli minimalnie różne przez zaokrąglenia, tnij do wspólnego minimalnego rozmiaru
        X = min(vol_top.shape[0], vol_side.shape[0])
        Y = min(vol_top.shape[1], vol_side.shape[1])
        Z = min(vol_top.shape[2], vol_side.shape[2])
        vol_top = vol_top[:X, :Y, :Z]
        vol_side = vol_side[:X, :Y, :Z]

    vol_intersection = np.logical_and(vol_top, vol_side)

    print(f"[3D] DIAMETER_MM={DIAMETER_MM} mm")
    print(f"[3D] voxel_mm={voxel_mm_top:.6f} mm/voxel  |  mm_per_px={mm_per_px_top:.6f} mm/px")
    print(f"[3D] vol_top filled = {int(vol_top.sum())}")
    print(f"[3D] vol_side filled = {int(vol_side.sum())}")
    print(f"[3D] intersection filled = {int(vol_intersection.sum())}  <-- WYPADKOWY BĄBEL")

    show_3d_volume_scatter(vol_intersection, "KROK 4: Część wspólna rzutów (wypadkowy bąbel)", max_points=120000)

    # ============================================================
    # KROK 5: szybki export PLY (chmura punktów)
    # ============================================================
    out_ply = "bubble_intersection.ply"
    export_volume_to_ply_pointcloud(vol_intersection, out_ply, voxel_mm=voxel_mm_top, center_yz=True, max_points=1_000_000)


if __name__ == "__main__":
    main()