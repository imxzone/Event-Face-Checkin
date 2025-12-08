import os
import cv2
from glob import glob
from tqdm import tqdm
from src.config import Config
from src.face_model import init_insightface


def resize_if_large(img, max_size=None):
    if max_size is None:
        max_size = Config.image_max_size

    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img, 1.0

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    return img_resized, scale


def crop_main_face(app, img_path, margin_ratio=None):
    if margin_ratio is None:
        margin_ratio = Config.crop_margin_ratio

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Không đọc được ảnh: {img_path}")
        return None

    h_orig, w_orig = img.shape[:2]
    img_resized, scale = resize_if_large(img)
    faces = app.get(img_resized)

    if len(faces) == 0:
        print(f"[WARN] Không tìm thấy mặt trong ảnh: {os.path.basename(img_path)}")
        return None

    # chọn mặt lớn nhất
    faces = sorted(
        faces,
        key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
        reverse=True,
    )
    main_face = faces[0]
    x1, y1, x2, y2 = main_face.bbox

    x1 = int(x1 / scale)
    y1 = int(y1 / scale)
    x2 = int(x2 / scale)
    y2 = int(y2 / scale)

    w_box = x2 - x1
    h_box = y2 - y1
    mx = int(w_box * margin_ratio)
    my = int(h_box * margin_ratio)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w_orig, x2 + mx)
    y2 = min(h_orig, y2 + my)

    face_crop = img[y1:y2, x1:x2].copy()

    if face_crop.size == 0:
        print(f"[WARN] Crop lỗi (empty): {os.path.basename(img_path)}")
        return None

    face_crop = cv2.resize(face_crop, Config.output_face_size)
    return face_crop


def preprocess_all_images():
    app = init_insightface()

    raw_dir = Config.raw_image_dir
    face_dir = Config.face_dir
    os.makedirs(face_dir, exist_ok=True)

    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        img_paths.extend(glob(os.path.join(raw_dir, ext)))

    if not img_paths:
        print(f"[WARN] Không tìm thấy ảnh nào trong {raw_dir}")
        return

    print(f"[INFO] Bắt đầu preprocess {len(img_paths)} ảnh...\n")

    saved_count = 0
    failed_images = []

    for img_path in tqdm(img_paths, desc="Đang xử lý ảnh", ncols=100):
        fname = os.path.basename(img_path)
        out_path = os.path.join(face_dir, fname)

        face_img = crop_main_face(app, img_path)

        if face_img is None:
            failed_images.append(fname)
            continue

        cv2.imwrite(out_path, face_img)
        saved_count += 1

    print("\n[INFO] Hoàn tất preprocess!")
    print(f" - Thành công: {saved_count}/{len(img_paths)} ảnh")
    print(f" - Thất bại:   {len(failed_images)} ảnh")

    if failed_images:
        print("\n[WARN] Các ảnh lỗi:")
        for name in failed_images:
            print("  -", name)
