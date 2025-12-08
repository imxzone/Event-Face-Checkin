import json

def export_user_json(users, output_file="checkedin_users.json"):
    data = {
            "users": users
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"[INFO] Đã xuất JSON: {output_file}")
