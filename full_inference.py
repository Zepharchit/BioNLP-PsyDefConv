# ============================================================================
# final_inference.py
# Combines Stage 1 gate (SVM) + Stage 2 LLM → final predictions
# Run after: stage1_binary_clf.py + lora_training.py / qlora_training.py
#
# Usage:
#   python final_inference.py --model lora            # 3-fold CV ensemble
#   python final_inference.py --model qlora           # 3-fold CV ensemble
#   python final_inference.py --model lora  --single  # single-train run
#   python final_inference.py --model qlora --single  # single-train run
# ============================================================================

import argparse
import joblib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.special import softmax
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support,
)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["lora", "qlora"], default="lora",
                    help="Which Stage 2 model to load (default: lora)")
parser.add_argument("--single", action="store_true",
                    help="Load single-train logits instead of 3-fold ensemble")
args = parser.parse_args()

# ── Config ───────────────────────────────────────────────────────────────────
STAGE2_DIRS = {
    "lora":  Path("/root/output/stage2_lora"),
    "qlora": Path("/root/output/stage2_qlora"),
}
OUTPUT_DIR = STAGE2_DIRS[args.model]
GATE_DIR   = Path("/root/output")
N_FOLDS    = 3

ALL_LABELS = list(range(9))
POS_LABELS = list(range(1, 9))

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
gate = gate_clf.predict(X_te)
print(f"Gate → predicted class 0: {(gate==0).sum()}  non-zero: {(gate==1).sum()}")

# ── Stage 2: load logits ──────────────────────────────────────────────────────
if args.single:
    logit_path = OUTPUT_DIR / "single" / "test_logits.npy"
    if not logit_path.exists():
        raise FileNotFoundError(
            f"No logits at {logit_path}. Run with --single flag first."
        )
    test_logits = np.load(logit_path)
    print(f"\nLoaded single-train logits: {logit_path}  shape={test_logits.shape}")
else:
    print(f"\nLoading Stage 2 logits from {N_FOLDS} folds...")
    fold_logits = []
    for fold_idx in range(N_FOLDS):
        logit_path = OUTPUT_DIR / f"fold_{fold_idx}" / "test_logits.npy"
        if not logit_path.exists():
            raise FileNotFoundError(
                f"No logits at {logit_path}. Run training first."
            )
        fold_logits.append(np.load(logit_path))
        print(f"  fold {fold_idx}: shape={fold_logits[-1].shape}")
    test_logits = sum(fold_logits) / N_FOLDS

# softmax → argmax → remap 0-7 back to 1-8
test_ens  = softmax(test_logits, axis=1)
llm_preds = test_ens.argmax(axis=1) + 1

# ── Combine Stage 1 + Stage 2 ─────────────────────────────────────────────────
final = np.where(gate == 0, 0, llm_preds)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    out = {}
    for tag, labs in (("all", ALL_LABELS), ("pos", POS_LABELS)):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labs, average="macro", zero_division=0,
        )
        out[f"p_{tag}"]  = p
        out[f"r_{tag}"]  = r
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


run_tag = f"{args.model}_{'single' if args.single else '3fold'}"
print_metrics(compute_metrics(y_true, llm_preds), f"Stage 2 {args.model} only (no gate)")
print_metrics(compute_metrics(y_true, final),     f"Final (Stage 1 gate + Stage 2 {args.model})")

print("\n--- Full classification report ---")
print(classification_report(y_true, final, labels=ALL_LABELS, zero_division=0))

# ── Confusion matrix ──────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, preds, title in zip(
    axes,
    [llm_preds, final],
    [f"Stage 2 {args.model} only", f"Stage 1 gate + Stage 2 {args.model}"],
):
    cm = confusion_matrix(y_true, preds, labels=ALL_LABELS)
    ConfusionMatrixDisplay(cm, display_labels=ALL_LABELS).plot(ax=ax, cmap="Blues")
    ax.set_title(title)
plt.tight_layout()
cm_path = OUTPUT_DIR / f"final_cm_{run_tag}.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\nConfusion matrix → {cm_path}")

# ── Metrics CSV ───────────────────────────────────────────────────────────────
csv_path = OUTPUT_DIR / f"metrics_{run_tag}.csv"
rows = [
    {"split": "test_llm_only", "model": args.model, "single": args.single, "gate": False,
     **compute_metrics(y_true, llm_preds)},
    {"split": "test_final",    "model": args.model, "single": args.single, "gate": True,
     **compute_metrics(y_true, final)},
]
df_metrics = pd.DataFrame(rows)
if csv_path.exists():
    df_metrics.to_csv(csv_path, mode="a", header=False, index=False)
else:
    df_metrics.to_csv(csv_path, mode="w", header=True, index=False)
print(f"Metrics CSV → {csv_path}")

# ── Save final predictions ────────────────────────────────────────────────────
pred_path = OUTPUT_DIR / f"final_predictions_{run_tag}.csv"
test_df["pred_llm_only"] = llm_preds
test_df["pred_gate"]     = gate
test_df["pred_final"]    = final
test_df.to_csv(pred_path, index=False)
print(f"Predictions CSV → {pred_path}")
print(f"\nBaseline to beat (Na et al., Ministral-8B): pos=0.3148  all=0.3745")
