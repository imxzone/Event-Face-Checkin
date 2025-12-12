from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
import base64
import cv2
import numpy as np
from src.face_model import init_insightface, extract_faces, extract_embedding
from src.database import load_database, find_best_match

app = Flask(__name__)
# CORS configuration for cross-origin requests (2 máy khác nhau)
# This allows requests from any origin (frontend on different machine)
CORS(app, resources={r"/*": {"origins": "*"}})

# Explicit handler for OPTIONS preflight requests - CRITICAL for dev tunnels
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        response.headers.add("Access-Control-Max-Age", "3600")
        return response

# Ensure CORS headers on ALL responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
    return response

JSON_FILE = os.path.join(os.path.dirname(__file__), "checkedin_users.json")

# Initialize AI model and database once at startup
face_app = None
db_embeddings = None
db_labels = None

try:
    print("[INFO] Initializing InsightFace model...")
    face_app = init_insightface()
    if face_app is None:
        raise Exception("Failed to initialize InsightFace model")
    print("[INFO] ✓ InsightFace model initialized")
    
    print("[INFO] Loading face database...")
    db_embeddings, db_labels = load_database()
    if db_embeddings is None or db_labels is None:
        raise Exception("Failed to load database")
    print(f"[INFO] ✓ Database loaded: {len(db_labels)} faces")
    print("[INFO] ✓ API ready!")
except FileNotFoundError as e:
    print(f"[ERROR] File not found: {e}")
    print("[WARN] Please run: python build_face_db.py")
    import traceback
    traceback.print_exc()
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("[WARN] Please install dependencies: pip install -r requirements.txt")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"[ERROR] Failed to initialize: {e}")
    import traceback
    traceback.print_exc()
    print("[WARN] API will not work until database is built. Run build_face_db.py first.")


def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        print(f"[ERROR] Failed to decode base64 image: {e}")
        return None


@app.route("/api/users", methods=["GET"])
def get_users():
    if not os.path.exists(JSON_FILE):
        return jsonify({"users": []}), 200

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return jsonify(data), 200


@app.route("/recognize", methods=["POST", "OPTIONS"])
def recognize_face():
    # Handle preflight OPTIONS request explicitly
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        return response, 200
    """
    Recognize faces from base64 image
    Request body: { "image": "base64_string" }
    Response: { "users": [{ "userId": "...", "name": "...", "confidence": 0.95 }, ...] }
    """
    if face_app is None or db_embeddings is None or db_labels is None:
        return jsonify({
            "error": "AI model or database not loaded. Please build the database first.",
            "users": []
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "error": "Missing 'image' field in request body",
                "users": []
            }), 400
        
        base64_image = data['image']
        
        frame = base64_to_image(base64_image)
        if frame is None:
            return jsonify({
                "error": "Failed to decode image",
                "users": []
            }), 400
        
        faces = extract_faces(face_app, frame)
        
        if len(faces) == 0:
            return jsonify({
                "users": [],
                "message": "No faces detected"
            }), 200
        
        recognized_users = []
        
        for face in faces:
            embedding = extract_embedding(face)
            
            match_label, similarity = find_best_match(
                embedding, db_embeddings, db_labels
            )
            
            if match_label is not None and similarity >= 0.4:  # Use threshold from config
                recognized_users.append({
                    "userId": match_label["ID_Name"],
                    "name": match_label["Name"],
                    "confidence": float(similarity),
                    "house": match_label.get("House", "")
                })
        
        recognized_users.sort(key=lambda x: x["confidence"], reverse=True)
        
        return jsonify({
            "users": recognized_users,
            "detected_faces": len(faces),
            "recognized_faces": len(recognized_users)
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Recognition error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "users": []
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "model_loaded": face_app is not None,
        "database_size": len(db_labels) if db_labels is not None else 0
    }), 200


if __name__ == "__main__":
    import sys
    
    if face_app is None or db_embeddings is None or db_labels is None:
        print("\n" + "=" * 60)
        print("ERROR: Cannot start API server!")
        print("=" * 60)
        print("\nPlease ensure:")
        print("1. Face database has been built (run: python build_face_db.py)")
        print("2. All required files exist in data/processed/")
        print("3. All dependencies are installed (run: pip install -r requirements.txt)")
        print("\n" + "=" * 60)
        sys.exit(1)
    
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("\n" + "=" * 60)
    print("Starting Face Recognition API Server")
    print("=" * 60)
    print(f"Server: http://0.0.0.0:{port}")
    print(f"Health check: http://0.0.0.0:{port}/health")
    print(f"Recognize endpoint: http://0.0.0.0:{port}/recognize")
    print(f"Database: {len(db_labels)} faces loaded")
    print(f"Debug mode: {debug}")
    print("=" * 60 + "\n")
    
    try:
        app.run(host="0.0.0.0", port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n\n[INFO] Server stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
