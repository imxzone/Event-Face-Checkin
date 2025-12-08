import cv2
from datetime import datetime
from collections import defaultdict

from src.config import Config
from src.face_model import init_insightface, extract_faces, extract_embedding
from src.database import load_database, find_best_match
from src.checkin import init_checkin_log, draw_result, log_checkin

def main():
    app = init_insightface()
    db_embeddings, db_labels = load_database()
    pending_frames = defaultdict(int)
    was_in_frame = defaultdict(bool)
    last_left_time = defaultdict(lambda: None)
    last_checkin_time = defaultdict(lambda: None)
    session_checked_in = defaultdict(bool)

    init_checkin_log()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không mở được webcam.")
        return

    print("[INFO] Webcam ready - Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Webcam lost.")
            break

        now = datetime.now()
        display_frame = frame.copy()
        faces = extract_faces(app, frame)
        in_frame_now = set()

        for face in faces:
            embedding = extract_embedding(face)
            match_label, similarity = find_best_match(
                embedding, db_embeddings, db_labels
            )

            confirmed = False
            pid = None
            name = None
            house = None

            if match_label is not None:
                pid = match_label["ID_Name"]
                name = match_label["Name"]
                house = match_label["House"]

                in_frame_now.add(pid)
                pending_frames[pid] = min(
                    pending_frames.get(pid, 0) + 1,
                    Config.confirm_frames
                )

                if pending_frames[pid] >= Config.confirm_frames:
                    confirmed = True
            else:
                confirmed = False
            draw_result(display_frame, face, match_label, similarity, confirmed)

            if confirmed and pid is not None:
                allow = False
                first_time = last_checkin_time[pid] is None

                if first_time:
                    allow = True
                else:
                    left_time = last_left_time[pid]
                    if left_time is not None:
                        gap = (now - left_time).total_seconds()
                        if gap >= Config.checkin_cooldown and not session_checked_in[pid]:
                            allow = True

                if allow:
                    log_checkin(pid, name, house, similarity)
                    last_checkin_time[pid] = now
                    session_checked_in[pid] = True

        all_ids = set(list(was_in_frame.keys()) + list(in_frame_now))

        for pid in all_ids:
            if was_in_frame[pid] and pid not in in_frame_now:
                last_left_time[pid] = now
                pending_frames[pid] = 0
                session_checked_in[pid] = False 
            was_in_frame[pid] = (pid in in_frame_now)

        cv2.imshow("Face Check-in", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
