import os
import cv2
import pandas as pd
from datetime import datetime
from src.config import Config
from src.export_json import export_user_json 

checked_in_users = []


def init_checkin_log():
    os.makedirs(Config.processed_dir, exist_ok=True)
    if not os.path.exists(Config.checkin_log):
        with open(Config.checkin_log, "w", encoding="utf-8-sig") as f:
            f.write("timestamp,ID_Name,Name,House,similarity\n")


def _update_guest_sheet(name, house, ts_iso):
    if not os.path.exists(Config.guest_csv_file):
        print(f"[WARN] guest_csv_file không tồn tại: {Config.guest_csv_file}")
        return

    df = pd.read_csv(Config.guest_csv_file)

    if "Name" not in df.columns or "House" not in df.columns:
        print("[WARN] guest_csv_file không có cột Name/House")
        return

    # Tìm dòng tương ứng với khách
    mask = (df["Name"] == name) & (df["House"] == house)

    if not mask.any():
        print(f"[WARN] Không tìm thấy dòng tương ứng trong guest CSV: {name} - {house}")
        return

    idx = df.index[mask][0]

    # checkin_time: chỉ ghi lần đầu
    if "checkin_time" in df.columns:
        current_first = df.at[idx, "checkin_time"]
        if pd.isna(current_first) or str(current_first).strip() == "":
            df.at[idx, "checkin_time"] = ts_iso

    # checkin_last: luôn cập nhật lần gần nhất
    if "checkin_last" in df.columns:
        df.at[idx, "checkin_last"] = ts_iso

    # checkin_count: tăng thêm 1
    if "checkin_count" in df.columns:
        current_count = df.at[idx, "checkin_count"]
        try:
            if pd.isna(current_count) or str(current_count).strip() == "":
                count_val = 0
            else:
                count_val = int(current_count)
        except Exception:
            count_val = 0

        df.at[idx, "checkin_count"] = count_val + 1

    df.to_csv(Config.guest_csv_file, index=False, encoding="utf-8-sig")
    print(f"[INFO] Đã cập nhật guest CSV cho: {name} - {house}")


def log_checkin(pid, name, house, similarity):
    global checked_in_users

    ts = datetime.now().isoformat(timespec="seconds")

    # 1) Log vào file checkins.csv (log chi tiết)
    with open(Config.checkin_log, "a", encoding="utf-8-sig") as f:
        f.write(f"{ts},{pid},{name},{house},{similarity:.4f}\n")

    # 2) Cập nhật vào file guest CSV chính
    _update_guest_sheet(name, house, ts)

    print(f"[CHECK-IN] {ts} - {pid} - {name} ({house}) sim={similarity:.4f}")
    already_exists = any(u["userId"] == pid for u in checked_in_users)
    if already_exists:
        return
    checked_in_users.append({
        "userId": pid,
        "name": name
    })

    export_user_json(checked_in_users)


def draw_result(display_frame, face, match_label, similarity, confirmed: bool):
    """
    Vẽ khung và text lên frame:
        - Khung vàng khi đang xác nhận
        - Khung xanh khi check-in OK
        - Khung đỏ khi Unknown
    """
    x1, y1, x2, y2 = face.bbox.astype(int)

    if match_label is not None:
        pid = match_label["ID_Name"]
        name = match_label["Name"]

        if not confirmed:
            # khung vàng: đang xác nhận
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                display_frame,
                f"{name} ({pid})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display_frame,
                "Dang xac nhan...",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        else:
            # khung xanh: xác nhận OK
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"{name} ({pid})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                "Check-in OK",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    else:
        # unknown face
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            display_frame,
            "Unknown",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
