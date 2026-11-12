# ============================================================================
# stage1_binary_clf.py
# Binary gate: No Defense (class 0) vs Has Defense (classes 1–8)
# Compares Logistic Regression, SVM, Random Forest on sentence embeddings
# ============================================================================

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ── Config ───────────────────────────────────────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"
VAL_SIZE     = 0.2
RANDOM_STATE = 421

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading dataset...")
ds       = load_dataset("AIMH/PsyDefConv")
train_df = ds["train"].to_pandas()
test_df  = ds["test"].to_pandas()
train_df["label"] = train_df["label"].astype(int)
test_df["label"]  = test_df["label"].astype(int)

print(f"Train: {len(train_df)}  Test: {len(test_df)}")
print(f"Train class 0: {(train_df['label']==0).sum()}  non-0: {(train_df['label']!=0).sum()}")
print(f"Test  class 0: {(test_df['label']==0).sum()}   non-0: {(test_df['label']!=0).sum()}")

# ── Binary targets ───────────────────────────────────────────────────────────
y_all  = (train_df["label"] != 0).astype(int).values  # 0=no defense, 1=has defense
y_test = (test_df["label"]  != 0).astype(int).values

# ── Train / val split ────────────────────────────────────────────────────────
X_text_train, X_text_val, y_tr, y_val = train_test_split(
    train_df["current_text"].tolist(), y_all,
    test_size=VAL_SIZE, stratify=y_all, random_state=RANDOM_STATE,
)
print(f"\nTrain split: {len(X_text_train)}  Val split: {len(X_text_val)}")

# ── Embed ────────────────────────────────────────────────────────────────────
print(f"\nEmbedding with {EMBED_MODEL}...")
embedder = SentenceTransformer(EMBED_MODEL)

X_tr  = embedder.encode(X_text_train, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
X_val = embedder.encode(X_text_val,   batch_size=64, show_progress_bar=True, normalize_embeddings=True)
X_te  = embedder.encode(test_df["current_text"].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
print(f"Embedding shape: {X_tr.shape}")

# ── Classifiers ──────────────────────────────────────────────────────────────
clfs = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
    "SVM (RBF)":          CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, class_weight="balanced"), ensemble=False),
    "RandomForest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
}

# ── Fit on train split, compare on val ───────────────────────────────────────
results = []
print(f"\n{'─'*50}")
print(f"{'Model':<25} {'Val Macro-F1':<12} {'Val Acc'}")
print(f"{'─'*50}")

for name, clf in clfs.items():
    clf.fit(X_tr, y_tr)
    val_preds = clf.predict(X_val)
    val_f1    = f1_score(y_val, val_preds, average="macro")
    val_acc   = (val_preds == y_val).mean()
    print(f"{name:<25} {val_f1:.4f}       {val_acc:.4f}")
    results.append({"model": name, "val_f1": val_f1, "clf": clf})

# ── Pick best by val F1, refit on full train, evaluate once on test ──────────
best = max(results, key=lambda x: x["val_f1"])
print(f"\nBest model: {best['model']} (val F1={best['val_f1']:.4f})")
print("Refitting on full training set...")
best["clf"].fit(
    embedder.encode(train_df["current_text"].tolist(), batch_size=64,
                    show_progress_bar=False, normalize_embeddings=True),
    y_all,
)

# ── Final evaluation on test ─────────────────────────────────────────────────
print(f"\n{'─'*50}")
print("Final evaluation on held-out test set")
print(f"{'─'*50}")
test_preds = best["clf"].predict(X_te)
print(classification_report(y_test, test_preds,
                             target_names=["No Defense (0)", "Has Defense (1-8)"]))

# ── Error analysis ────────────────────────────────────────────────────────────
errors = test_df[test_preds != y_test].copy()
print(f"Misclassified: {len(errors)} / {len(test_df)}")
print("Breakdown of misclassified true labels:")
print(errors["label"].value_counts().to_string())

# ── Save gate model ───────────────────────────────────────────────────────────
GATE_DIR = Path("/root/output")
GATE_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(best["clf"], GATE_DIR / "stage1_gate.pkl")
joblib.dump(embedder,    GATE_DIR / "stage1_embedder.pkl")
print(f"\nGate model → {GATE_DIR / 'stage1_gate.pkl'}")
print(f"Embedder   → {GATE_DIR / 'stage1_embedder.pkl'}")