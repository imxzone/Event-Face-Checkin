import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from src.config import Config
from src.face_model import init_insightface


def get_face_embedding(app, img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Không đọc được ảnh: {os.path.basename(img_path)}")
        return None

    faces = app.get(img)
    if len(faces) == 0:
        print(f"[WARN] Không tìm thấy mặt trong ảnh: {os.path.basename(img_path)}")
        return None

    # lấy mặt lớn nhất
    faces = sorted(
        faces,
        key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
        reverse=True,
    )
    main_face = faces[0]
    return main_face.normed_embedding


def build_face_database():
    os.makedirs(Config.processed_dir, exist_ok=True)

    app = init_insightface()
    df = pd.read_csv(Config.csv_info_file)

    all_embeddings = []
    labels = []
    failed_people = []

    print(f"[INFO] Bắt đầu build database cho {len(df)} người...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Đang tạo database", ncols=100):
        name = row["Name"]
        house = row["House"]
        person_id = row["ID_Name"]

        patterns = [
            os.path.join(Config.face_dir, f"{person_id}_*.jpg"),
            os.path.join(Config.face_dir, f"{person_id}_*.jpeg"),
            os.path.join(Config.face_dir, f"{person_id}_*.png"),
        ]

        img_paths = []
        for p in patterns:
            img_paths.extend(glob(p))

        if not img_paths:
            failed_people.append(person_id)
            print(f"[WARN] Không có ảnh cho {person_id} - {name}")
            continue

        person_embeddings = []

        for img_path in img_paths:
            emb = get_face_embedding(app, img_path)
            if emb is not None:
                person_embeddings.append(emb)

        if len(person_embeddings) == 0:
            failed_people.append(person_id)
            print(f"[WARN] Không tạo được embedding cho {person_id} - {name}")
            continue

        mean_emb = np.mean(np.stack(person_embeddings), axis=0)

        all_embeddings.append(mean_emb)
        labels.append({
            "ID_Name": person_id,
            "Name": name,
            "House": house,
        })

    if len(all_embeddings) == 0:
        print("[ERROR] Không có embedding nào được tạo!")
        return

    all_embeddings = np.stack(all_embeddings)
    labels_df = pd.DataFrame(labels)

    np.save(Config.embeddings_file, all_embeddings)
    labels_df.to_csv(Config.labels_file, index=False, encoding="utf-8-sig")

    print("\n[INFO] Hoàn tất build database!")
    print(f" - Thành công: {len(all_embeddings)} người")
    print(f" - Thất bại:   {len(failed_people)} người")

    if failed_people:
        print("\n[WARN] Danh sách ID lỗi:")
        for pid in failed_people:
            print("  -", pid)