from flask import Flask, jsonify
import json
import os

app = Flask(__name__)

JSON_FILE = os.path.join(os.path.dirname(__file__), "checkedin_users.json")


@app.route("/api/users", methods=["GET"])
def get_users():
    if not os.path.exists(JSON_FILE):
        return jsonify({"users": []}), 200

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return jsonify(data), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
