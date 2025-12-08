# multi_face_crop.py
import os
import cv2
from glob import glob
from tqdm import tqdm
from insightface.app import FaceAnalysis

# ============ CONFIG ============= #

RAW_DIR = "data/khachmoi"        # folder chứa ảnh nhiều người
OUT_DIR = "data/khachmoi_crop"      # folder output
FACE_SIZE = (512, 512)                # resize mặt về size chuẩn
FACE_MIN_SIZE = 60                    # lọc bỏ mặt quá nhỏ (pixels)
CONF_THRESHOLD = 0.4                 # confidence threshold
MARGIN_RATIO = 0.6                    # bo viền rộng hơn quanh mặt

# ================================= #

os.makedirs(OUT_DIR, exist_ok=True)


def init_detector():
    """Khởi tạo InsightFace detector."""
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU
    return app


def crop_all_faces(app, img_path):
    """Crop toàn bộ gương mặt trong ảnh."""
    img = cv2.imread(img_path)
    if img is None:
        return []  # không đọc được ảnh → coi như fail

    h, w = img.shape[:2]
    faces = app.get(img)

    if len(faces) == 0:
        return []  # không có mặt → fail

    cropped_list = []
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    for idx, face in enumerate(faces):

        # Lọc theo confidence
        if hasattr(face, "det_score") and face.det_score < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = face.bbox.astype(int)
        box_w = x2 - x1
        box_h = y2 - y1

        # Bỏ mặt nhỏ
        if box_w < FACE_MIN_SIZE or box_h < FACE_MIN_SIZE:
            continue

        # Thêm margin
        mx = int(box_w * MARGIN_RATIO)
        my = int(box_h * MARGIN_RATIO)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        face_img = img[y1:y2, x1:x2].copy()
        face_img = cv2.resize(face_img, FACE_SIZE)

        out_path = os.path.join(OUT_DIR, f"{base_name}_face_{idx+1}.jpg")
        cv2.imwrite(out_path, face_img)
        cropped_list.append(out_path)

    return cropped_list


def process_folder():
    app = init_detector()

    # load danh sách ảnh
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        img_paths.extend(glob(os.path.join(RAW_DIR, ext)))

    print(f"[INFO] Tổng số ảnh cần xử lý: {len(img_paths)}\n")

    failed_images = []  # lưu danh sách ảnh không crop được

    # Thanh loading với tqdm
    for img_path in tqdm(img_paths, desc="Đang crop ảnh", unit="ảnh"):
        outputs = crop_all_faces(app, img_path)

        if not outputs:
            failed_images.append(img_path)

    print("\n-----------------------------------------")
    print("[KẾT QUẢ] Quá trình crop hoàn tất.")
    print(f"- Số ảnh không crop được mặt: {len(failed_images)}")

    if failed_images:
        print("\nCác ảnh KHÔNG crop được:")
        for f in failed_images:
            print("  -", f)
    else:
        print("Tất cả ảnh đều crop được mặt.")
    print("-----------------------------------------")


if __name__ == "__main__":
    process_folder()
