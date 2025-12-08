import os

class Config:
    data_dir = "data"
    raw_image_dir = os.path.join(data_dir, "raw")
    face_dir = os.path.join(data_dir, "faces")
    processed_dir = os.path.join(data_dir, "processed")
    guest_csv_file = "guests.csv"
    csv_info_file = "data.csv"
    embeddings_file = os.path.join(processed_dir, "face_embeddings.npy")
    labels_file = os.path.join(processed_dir, "face_labels.csv")
    checkin_log = os.path.join(processed_dir, "checkins.csv")

    model_name = "buffalo_l"
    ctx_id = -1                  # -1 = CPU
    det_size = (640, 640)        # detect window size

    image_max_size = 1600
    crop_margin_ratio = 0.6
    output_face_size = (512, 512)

    confirm_frames = 5
    checkin_cooldown = 4
    threshold = 0.4
    min_face_size = 60

    show_fps = True
