from insightface.app import FaceAnalysis
from src.config import Config

def init_insightface():
    app = FaceAnalysis(name=Config.model_name)
    app.prepare(
        ctx_id=Config.ctx_id,
        det_size=Config.det_size
    )
    return app

def extract_faces(app, frame):
    return app.get(frame)

def extract_embedding(face):
    return face.normed_embedding