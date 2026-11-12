import joblib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support,
)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_KEY    = "llama31_8b"
OUTPUT_DIR   = Path("/workspace/output") / MODEL_KEY
GATE_DIR     = Path("/workspace/output")          # where gate pkl files are saved
SEED_WEIGHTS = [0.5, 0.5]                         # must match inference_only.py run

NUM_CLASSES = 9
ALL_LABELS  = list(range(NUM_CLASSES))
POS_LABELS  = list(range(1, NUM_CLASSES))

# ── Load test data ────────────────────────────────────────────────────────────
print("Loading dataset...")
ds      = load_dataset("AIMH/PsyDefConv")
test_df = ds["test"].to_pandas()
test_df["label"] = test_df["label"].astype(int)
y_true  = test_df["label"].values
print(f"Test samples: {len(test_df)}")

# ── Stage 1: gate ─────────────────────────────────────────────────────────────
print("\nRunning Stage 1 gate...")
embedder = joblib.load(GATE_DIR / "stage1_embedder.pkl")
gate_clf = joblib.load(GATE_DIR / "stage1_gate.pkl")

X_te = embedder.encode(
    test_df["current_text"].tolist(),
    batch_size=64, show_progress_bar=True, normalize_embeddings=True,
)
gate = gate_clf.predict(X_te)   # 0 = no defense, 1 = has defense
print(f"Gate → predicted class 0: {(gate==0).sum()}  non-zero: {(gate==1).sum()}")

# ── Stage 2: LLM ensemble probs ───────────────────────────────────────────────
tag       = f"w{'_'.join(str(w) for w in SEED_WEIGHTS)}"
ens_path  = OUTPUT_DIR / f"test_ensemble_{tag}.npy"

print(f"\nLoading LLM ensemble probs from {ens_path}...")
if not ens_path.exists():
    raise FileNotFoundError(
        f"No ensemble file found at {ens_path}. "
        "Run inference_only.py first."
    )
test_ens  = np.load(ens_path)            # (n_test, 9)
llm_preds = test_ens.argmax(axis=1)      # raw LLM predictions (1–8)

# ── Combine Stage 1 + Stage 2 ─────────────────────────────────────────────────
final = np.where(gate == 0, 0, llm_preds)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    out = {}
    for tag, labs in (("all", ALL_LABELS), ("pos", POS_LABELS)):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labs, average="macro", zero_division=0,
        )
        out[f"p_{tag}"] = p
        out[f"r_{tag}"] = r
        out[f"f1_{tag}"] = f
    return out

def print_metrics(metrics, title=""):
    if title:
        print(f"\n{'─'*50}\n{title}\n{'─'*50}")
    rows = [
        ("",          "Precision", "Recall", "F1"),
        ("0–8 (all)", f"{metrics['p_all']:.4f}", f"{metrics['r_all']:.4f}", f"{metrics['f1_all']:.4f}"),
        ("1–8 (pos)", f"{metrics['p_pos']:.4f}", f"{metrics['r_pos']:.4f}", f"{metrics['f1_pos']:.4f}"),
    ]
    for row in rows:
        print(f"  {row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]}")

# LLM-only baseline (no gate)
print_metrics(compute_metrics(y_true, llm_preds), "LLM only (no gate)")

# Stage 1 + Stage 2 combined
print_metrics(compute_metrics(y_true, final), "Final (Stage1 gate + LLM)")

print("\n--- Full classification report ---")
print(classification_report(y_true, final, labels=ALL_LABELS, zero_division=0))

# ── Confusion matrix ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, preds, title in zip(
    axes,
    [llm_preds, final],
    ["LLM only", "Stage1 gate + LLM"],
):
    cm = confusion_matrix(y_true, preds, labels=ALL_LABELS)
    ConfusionMatrixDisplay(cm, display_labels=ALL_LABELS).plot(ax=ax, cmap="Blues")
    ax.set_title(title)
plt.tight_layout()
cm_path = OUTPUT_DIR / f"final_cm_{MODEL_KEY}.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\nConfusion matrix → {cm_path}")

# ── Metrics CSV ───────────────────────────────────────────────────────────────
csv_path = OUTPUT_DIR / f"metrics_{MODEL_KEY}.csv"
rows = [
    {"split": "test_llm_only",    "model": MODEL_KEY, "gate": False,
     **compute_metrics(y_true, llm_preds)},
    {"split": "test_final",       "model": MODEL_KEY, "gate": True,
     **compute_metrics(y_true, final)},
]
df_metrics = pd.DataFrame(rows)
if csv_path.exists():
    df_metrics.to_csv(csv_path, mode="a", header=False, index=False)
else:
    df_metrics.to_csv(csv_path, mode="w", header=True, index=False)
print(f"Metrics CSV → {csv_path}")

# ── Save final predictions ────────────────────────────────────────────────────
pred_path = OUTPUT_DIR / f"final_predictions_{MODEL_KEY}.csv"
test_df["pred_llm_only"] = llm_preds
test_df["pred_gate"]     = gate
test_df["pred_final"]    = final
test_df.to_csv(pred_path, index=False)
print(f"Predictions CSV → {pred_path}")
