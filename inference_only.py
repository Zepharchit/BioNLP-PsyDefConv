# ============================================================================
# inference_only.py
# PsyDefConv — Seeded ensemble + final metrics
# Loads saved .npy probs (fast) or reloads LoRA checkpoints (slow).
# Run after train_only.py has completed for all SEEDS.
# ============================================================================

# ── 0. Imports + Drive ───────────────────────────────────────────────────────
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from datasets import Dataset, load_dataset
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold


# ── 1. Config ────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "llama31_8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "qwen3_4b":   "unsloth/Qwen3-4B-Instruct",
    "phi4_mini":  "unsloth/Phi-4-mini-reasoning",
}
MODEL_KEY  = "llama31_8b"          # ← match what was used in train_only.py
MODEL_NAME = MODEL_CONFIGS[MODEL_KEY]

OUTPUT_DIR = Path("/workspace/psydef_unsloth") / MODEL_KEY

N_FOLDS      = 3
SEEDS        = [42, 123]
# ← adjust weights without retraining; must sum to 1
SEED_WEIGHTS = [0.5, 0.5]

MAX_LENGTH  = 1024
NUM_CLASSES = 9
ALL_LABELS  = list(range(NUM_CLASSES))
POS_LABELS  = list(range(1, NUM_CLASSES))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Model  : {MODEL_KEY}")
print(f"Output : {OUTPUT_DIR}")
print(f"Weights: seed_42={SEED_WEIGHTS[0]}  seed_123={SEED_WEIGHTS[1]}")

# ── 2. Data utilities ─────────────────────────────────────────────────────────
def parse_dialogue(d):
    if d is None:
        return []
    if isinstance(d, str):
        d = json.loads(d)
    return list(d)


def context_to_text(dialogue, current_text):
    turns = parse_dialogue(dialogue)
    if turns and str(turns[-1]["text"]).strip() == str(current_text).strip():
        turns = turns[:-1]
    return "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in turns)


def build_prompt(row, tokenizer, is_training=False):
    text    = str(row["current_text"])
    context = context_to_text(row.get("dialogue"), text)

    static = (
        "You are a Defense Mechanism Rating Scale (DMRS) specialist. "
        "Examine the dialog & utterance carefully and select the single most "
        "appropriate defense tier. When multiple defenses seem plausible, choose "
        "the tier with the strongest supporting evidence; if evidence is weak or "
        "contradictory, default to `0` (No defense).\n"
        "\nAvailable labels (DMRS tiers):\n"
        "0 = No defense\n"
        "1 = Action Defense Level (Acting Out / Help-Rejecting Complaining / Passive Aggression)\n"
        "2 = Major Image-distorting Defense Level (Splitting / Projective Identification)\n"
        "3 = Disavowal Defense Level (Denial / Projection / Rationalization / Autistic Fantasy)\n"
        "4 = Minor Image-distorting Defense Level (Devaluation / Idealization / Omnipotence)\n"
        "5 = Neurotic Defense Level (Displacement / Dissociation / Reaction Formation / Repression)\n"
        "6 = Obsessional Defense Level (Intellectualization / Isolation of Affects / Undoing)\n"
        "7 = Highly Adaptive Defense Level (Affiliation / Altruism / Anticipation / Humor / "
        "Self-Assertion / Self-Observation / Sublimation / Suppression)\n"
        "8 = Need More Information\n"
        "\nReturn exactly one label: <0-8>"
    )

    target    = f"\n- Target utterance:\n{text}\n"
    ctx_block = ""
    if context:
        fixed  = len(tokenizer(
            static + target + "\nAssistant: label: ",
            add_special_tokens=False,
        )["input_ids"])
        budget = MAX_LENGTH - fixed - 48
        if budget > 0:
            ids = tokenizer(context, add_special_tokens=False)["input_ids"][-budget:]
            ctx_block = "\n- Dialogue context:\n" + tokenizer.decode(ids) + "\n"

    user_content = static + ctx_block + target
    messages = [{"role": "user", "content": user_content}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return user_content + "\nAssistant: label: "


def df_to_hf(df, tokenizer):
    prompts = [build_prompt(row, tokenizer) for _, row in df.iterrows()]
    return Dataset.from_dict({"text": prompts, "label": df["label"].tolist()})


def load_data():
    ds       = load_dataset("AIMH/PsyDefConv")
    train_df = ds["train"].to_pandas()
    test_df  = ds["test"].to_pandas()
    for df in (train_df, test_df):
        df["label"] = df["label"].astype(int)
    group_col = "dialogue_id" if "dialogue_id" in train_df.columns else None
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")
    return train_df, test_df, group_col


# ── 4. Inference (used only when from_checkpoints=True) ─────────────────────
def _get_label_token_ids(tokenizer):
    return [
        tokenizer(str(i), add_special_tokens=False)["input_ids"][0]
        for i in range(NUM_CLASSES)
    ]


def predict_label_probs(model, tokenizer, prompts, batch_size=16):
    from unsloth import FastLanguageModel
    label_ids = _get_label_token_ids(tokenizer)
    FastLanguageModel.for_inference(model)
    all_probs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc   = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_LENGTH,
        ).to("cuda")
        with torch.no_grad():
            logits_full = model(**enc).logits
        last_pos = enc["attention_mask"].sum(dim=1) - 1
        for j, pos in enumerate(last_pos):
            l9    = torch.stack([logits_full[j, pos, tid] for tid in label_ids])
            probs = torch.softmax(l9, dim=0).float().cpu().numpy()
            all_probs.append(probs)
    return np.array(all_probs, dtype=np.float32)


# ── 5. Metrics ────────────────────────────────────────────────────────────────
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
        print(f"\n{'─'*45}\n{title}\n{'─'*45}")
    rows = [
        ("",         "Precision", "Recall", "F1"),
        ("0–8 (all)", f"{metrics['p_all']:.4f}", f"{metrics['r_all']:.4f}", f"{metrics['f1_all']:.4f}"),
        ("1–8 (pos)", f"{metrics['p_pos']:.4f}", f"{metrics['r_pos']:.4f}", f"{metrics['f1_pos']:.4f}"),
    ]
    for row in rows:
        print(f"  {row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]}")


def save_report(y_true, y_pred, stem):
    path = OUTPUT_DIR / f"{stem}.txt"
    mask = y_true != 0
    with open(path, "w") as f:
        f.write(f"=== {stem} ===\n\n")
        f.write("--- All labels (0-8) ---\n")
        f.write(classification_report(y_true, y_pred, zero_division=0))
        f.write("\n--- Positive labels (1-8) ---\n")
        f.write(classification_report(
            y_true[mask], y_pred[mask], labels=POS_LABELS, zero_division=0,
        ))
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
    ConfusionMatrixDisplay(cm, display_labels=ALL_LABELS).plot(ax=ax, cmap="Blues")
    ax.set_title(stem)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"{stem}_cm.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Report → {path}")
    print(f"  Confusion matrix → {fig_path}")


def append_metrics_csv(row: dict):
    csv_path = OUTPUT_DIR / f"metrics_{MODEL_KEY}.csv"
    df_row   = pd.DataFrame([row])
    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)
    print(f"  Metrics CSV → {csv_path}")


# ── 6. Persistence ────────────────────────────────────────────────────────────
def try_load_probs(seed, fold, split):
    p = OUTPUT_DIR / f"seed_{seed}" / f"fold_{fold}" / f"{split}_probs.npy"
    return np.load(p) if p.exists() else None

def load_val_indices(seed, fold):
    return np.load(OUTPUT_DIR / f"seed_{seed}" / f"fold_{fold}" / "val_indices.npy")


# ── 7. Ensemble ───────────────────────────────────────────────────────────────
def run_ensemble(seed_weights=None, from_checkpoints=False):
    """
    Fast path (default): loads .npy probs saved by train_only.py.
    Slow path (from_checkpoints=True): reloads LoRA checkpoints and re-runs inference.
    """
    if seed_weights is None:
        seed_weights = SEED_WEIGHTS
    assert len(seed_weights) == len(SEEDS) and abs(sum(seed_weights) - 1) < 1e-6

    train_df, test_df, group_col = load_data()

    oof_acc  = {}
    test_acc = {}

    for seed in SEEDS:
        # ── fast path: load saved .npy ────────────────────────────────────────
        oof_path  = OUTPUT_DIR / f"oof_probs_seed{seed}.npy"
        test_path = OUTPUT_DIR / f"test_probs_seed{seed}.npy"

        if not from_checkpoints and oof_path.exists() and test_path.exists():
            oof_acc[seed]  = np.load(oof_path)
            test_acc[seed] = np.load(test_path)
            print(f"  seed {seed}: loaded from {oof_path.name} / {test_path.name}")
            continue

        # ── slow path: reload LoRA checkpoints ───────────────────────────────
        from unsloth import FastLanguageModel
        groups  = (
            train_df[group_col].values if group_col else np.arange(len(train_df))
        )
        splits_ = list(
            StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            .split(train_df, train_df["label"], groups=groups)
        )

        oof_probs = np.zeros((len(train_df), NUM_CLASSES), dtype=np.float32)
        fold_test = []

        for fold, (_, val_idx) in enumerate(splits_):
            ckpt_dir    = OUTPUT_DIR / f"seed_{seed}" / f"fold_{fold}"
            checkpoints = sorted(ckpt_dir.glob("checkpoint-*"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
            ckpt = checkpoints[-1]
            print(f"  Loading {ckpt}")

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name     = str(ckpt),
                max_seq_length = MAX_LENGTH,
                dtype          = None,
                load_in_4bit   = True,
                device_map     = {"": 0},
            )
            val_df  = train_df.iloc[val_idx].reset_index(drop=True)
            val_ds  = df_to_hf(val_df,  tokenizer)
            test_ds = df_to_hf(test_df, tokenizer)

            val_probs  = predict_label_probs(model, tokenizer, val_ds["text"])
            t_probs    = predict_label_probs(model, tokenizer, test_ds["text"])
            oof_probs[val_idx] = val_probs
            fold_test.append(t_probs)

            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        oof_acc[seed]  = oof_probs
        test_acc[seed] = np.mean(fold_test, axis=0)

    # ── weighted ensemble ─────────────────────────────────────────────────────
    oof_ens  = sum(w * oof_acc[s]  for w, s in zip(seed_weights, SEEDS))
    test_ens = sum(w * test_acc[s] for w, s in zip(seed_weights, SEEDS))

    oof_preds  = oof_ens.argmax(axis=1)
    test_preds = test_ens.argmax(axis=1)

    tag = f"w{'_'.join(str(w) for w in seed_weights)}"
    print_metrics(compute_metrics(train_df["label"].values, oof_preds),
                  f"OOF  [{MODEL_KEY}] weights={seed_weights}")
    print_metrics(compute_metrics(test_df["label"].values,  test_preds),
                  f"Test [{MODEL_KEY}] weights={seed_weights}")

    save_report(train_df["label"].values, oof_preds,  f"oof_{MODEL_KEY}_{tag}")
    save_report(test_df["label"].values,  test_preds, f"test_{MODEL_KEY}_{tag}")

    append_metrics_csv({"split": "oof_ensemble",  "seed": "ensemble", "fold": "all",
                        "weights": str(seed_weights),
                        **compute_metrics(train_df["label"].values, oof_preds)})
    append_metrics_csv({"split": "test_ensemble", "seed": "ensemble", "fold": "all",
                        "weights": str(seed_weights),
                        **compute_metrics(test_df["label"].values, test_preds)})

    np.save(OUTPUT_DIR / f"oof_ensemble_{tag}.npy",  oof_ens)
    np.save(OUTPUT_DIR / f"test_ensemble_{tag}.npy", test_ens)
    print(f"\nSaved ensemble probs → {OUTPUT_DIR}")


# ── Run ───────────────────────────────────────────────────────────────────────
run_ensemble()

# ── Try a different weight split without retraining ───────────────────────────
# run_ensemble(seed_weights=[0.3, 0.7])
# run_ensemble(from_checkpoints=True)   # reload from saved LoRA checkpoints
