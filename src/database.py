import numpy as np
import pandas as pd
from src.config import Config

def load_database():
    embeddings = np.load(Config.embeddings_file)
    labels_df = pd.read_csv(Config.labels_file)
    return embeddings, labels_df

def cosine_similarity(vec, mat):
    vec = vec / np.linalg.norm(vec)
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    return mat @ vec

def find_best_match(embedding, db_embeddings, db_labels, threshold=None):
    if threshold is None:
        threshold = Config.threshold

    sims = cosine_similarity(embedding, db_embeddings)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim >= threshold:
        return db_labels.iloc[best_idx], best_sim
    else:
        return None, best_sim