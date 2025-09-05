import sys, os, time
from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis

snap = Path("media/snapshots/1.jpg")
if not snap.exists():
    sys.exit("No snapshot at media/snapshots/1.jpg. Run the runner or save a test image there.")

# Load InsightFace
app = FaceAnalysis(name='buffalo_l')  # or your model alias if different
app.prepare(ctx_id=0, det_size=(800, 800))  # ctx_id=-1 for CPU if needed

img = cv2.imread(str(snap))
if img is None:
    sys.exit("Failed to read snapshot.")

faces = app.get(img)
print(f"Detected: {len(faces)} face(s)")
if faces:
    emb = faces[0].normed_embedding
    print("Embedding dim:", emb.shape, "L2:", np.linalg.norm(emb))
