"""
train_rps_classifier.py – Train an MLP on MediaPipe‑hand landmarks
=================================================================
Assumes the following folder structure:
 data/
   rock/*.npy
   paper/*.npy
   scissors/*.npy
Produces:
 • rps_landmarks.joblib – scikit‑learn model
 • label_map.txt       – mapping index→class
"""

import glob
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ─────────────── load data ───────────────
CLASSES = ["rock", "paper", "scissors"]
X, y = [], []
for idx, cls in enumerate(CLASSES):
    for file in glob.glob(f"data/{cls}/*.npy"):
        X.append(np.load(file))
        y.append(idx)
X = np.vstack(X)
y = np.array(y)
print("Loaded:", {c: sum(y==i) for i, c in enumerate(CLASSES)})

# ─────────────── train / test split ───────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ─────────────── model ───────────────
clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
clf.fit(X_train, y_train)
print("Training accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))
print(classification_report(y_test, clf.predict(X_test), target_names=CLASSES))
print(confusion_matrix(y_test, clf.predict(X_test)))

# ─────────────── save model & labels ───────────────
joblib.dump(clf, "rps_landmarks.joblib")
with open("label_map.txt", "w") as f:
    f.write("\n".join(CLASSES))
print("✅ Model saved → rps_landmarks.joblib")
