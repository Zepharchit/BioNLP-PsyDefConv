# ============================================================================
# unsloth_training_script.py
# PsyDefConv — DMRS classification via Unsloth LLM SFT
# CV : StratifiedGroupKFold (N_FOLDS=3), two seeds → seeded ensemble
# ============================================================================

# ── 0. Imports + Drive ───────────────────────────────────────────────────────
import gc
import json
import re
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
from transformers import set_seed
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

# ── 1. Config ────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "llama31_8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "qwen3_4b":   "unsloth/Qwen3-4B-Instruct",
    "phi4_mini":  "unsloth/Phi-4-mini-reasoning",
}
MODEL_KEY  = "llama31_8b"          # ← change per experiment
MODEL_NAME = MODEL_CONFIGS[MODEL_KEY]

OUTPUT_DIR = Path("/workspace/outputs") / MODEL_KEY
AUG_PATH   = Path("/worspace/augmented_samples")
AUG_FILES  = []   # back-translation disabled; hesitation runs inline per fold

N_FOLDS      = 3
SEEDS        = [42, 123]
# Weights for seed_42 and seed_123 respectively — must sum to 1.
# E.g. [0.3, 0.7] to trust seed_123 more.
SEED_WEIGHTS = [0.5, 0.5]

MAX_LENGTH   = 1024
EPOCHS       = 1
LR           = 2e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE   = 8
GRAD_ACCUM   = 4
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

NUM_CLASSES     = 9
ALL_LABELS      = list(range(NUM_CLASSES))
POS_LABELS      = list(range(1, NUM_CLASSES))
MINORITY_LABELS = [1, 2, 3, 4, 5, 8]
TARGET_COUNT    = 200

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Model : {MODEL_NAME}")
print(f"Output: {OUTPUT_DIR}")

# ── 2. Data utilities ────────────────────────────────────────────────────────
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


def build_prompt(row, tokenizer, is_training=True):
    text    = str(row["current_text"])
    label   = int(row["label"])
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
    if is_training:
        messages.append({"role": "assistant", "content": f"label: {label}"})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=not is_training
        )
    prompt = user_content + "\nAssistant: label: "
    return (prompt + str(label)) if is_training else prompt


def df_to_hf(df, tokenizer, is_training=True):
    prompts = [build_prompt(row, tokenizer, is_training) for _, row in df.iterrows()]
    return Dataset.from_dict({"text": prompts, "label": df["label"].tolist()})


def load_data():
    ds       = load_dataset("AIMH/PsyDefConv")
    train_df = ds["train"].to_pandas()
    test_df  = ds["test"].to_pandas()
    for df in (train_df, test_df):
        df["label"] = df["label"].astype(int)
    group_col = "dialogue_id" if "dialogue_id" in train_df.columns else None
    print(f"Train: {len(train_df)}  Test: {len(test_df)}  Group: {group_col}")
    print(f"Label dist:\n{train_df['label'].value_counts().sort_index().to_dict()}")
    return train_df, test_df, group_col


def load_aug_data():
    if not AUG_FILES:
        return None
    dfs = []
    for fname in AUG_FILES:
        df = pd.read_parquet(AUG_PATH / fname)
        df["label"] = df["label"].astype(int)
        dfs.append(df)
        print(f"  Aug {fname}: {len(df)} rows")
    return pd.concat(dfs, ignore_index=True)


def augment_fold(tr_df):
    """
    Hesitation augmentation on one fold's training data only — never touches val/test.
    Round-robin over 3 modes so each augmented sample gets a distinct transform.
    Returns only the new augmented rows (not combined with tr_df).
    """
    CONTRACTIONS = {
        "I am": "I'm",   "do not": "don't",    "did not": "didn't",
        "can not": "can't", "cannot": "can't",  "will not": "won't",
        "would not": "wouldn't", "I have": "I've", "I would": "I'd",
        "I will": "I'll", "they are": "they're", "we are": "we're",
    }
    HEDGES      = ["Honestly, ", "Actually, ", "Personally, ", "To be honest, "]
    SHIFTS      = {"maybe": "perhaps", "like": "such as",
                   "kind of": "somewhat", "I think": "I believe"}
    FILLERS     = [" I guess.", ", you know.", ", I suppose."]
    MARKERS_MID = [", um,", ", uh,", ", you know,", ", I mean,", ", like,"]

    def apply_mode_0(text):
        for exp, con in CONTRACTIONS.items():
            if exp in text:
                text = text.replace(exp, con, 1)
                break
        hedge = HEDGES[hash(text) % len(HEDGES)]
        if not any(text.startswith(h) for h in HEDGES):
            text = hedge + text[0].lower() + text[1:]
        return text

    def apply_mode_1(text):
        for src, tgt in SHIFTS.items():
            if src in text:
                text = text.replace(src, tgt, 1)
                break
        if not any(text.endswith(f.strip()) for f in FILLERS):
            text = text.rstrip(".!?") + FILLERS[hash(text) % len(FILLERS)]
        return text

    def apply_mode_2(text):
        words = text.split()
        if len(words) < 5:
            return text
        pos = max(2, len(words) // 3)
        words.insert(pos, MARKERS_MID[hash(text) % len(MARKERS_MID)])
        return " ".join(words)

    modes = [apply_mode_0, apply_mode_1, apply_mode_2]
    pool  = tr_df[tr_df["label"].isin(MINORITY_LABELS)].copy()
    rows  = []

    for label, group in pool.groupby("label"):
        n       = max(1, TARGET_COUNT - len(group))
        samples = group.sample(n, replace=True, random_state=42).copy()
        samples["current_text"] = [
            modes[i % 3](text)
            for i, text in enumerate(samples["current_text"].tolist())
        ]
        rows.append(samples)
        print(f"    augment_fold label {label}: +{n} hes rows")

    if not rows:
        return pd.DataFrame(columns=tr_df.columns)

    aug = pd.concat(rows).reset_index(drop=True)
    aug["dialogue_id"] = aug["dialogue_id"] + "_hes"
    return aug


# ── 4. Model ─────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(seed):
    set_seed(seed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_LENGTH,
        dtype          = None,
        load_in_4bit   = True,
        device_map     = {"": 0},
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = LORA_R,
        target_modules             = ["q_proj","k_proj","v_proj","o_proj",
                                      "gate_proj","up_proj","down_proj"],
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = LORA_DROPOUT,
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = seed,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


# ── 5. Inference — next-token probabilities over digits 0–8 ─────────────────
def get_label_token_ids(tokenizer):
    return [
        tokenizer(str(i), add_special_tokens=False)["input_ids"][0]
        for i in range(NUM_CLASSES)
    ]


def predict_label_probs(model, tokenizer, prompts, batch_size=16):
    """
    Returns (N, 9) float32 array.
    Extracts next-token logits at the last input position, restricted to
    the digit tokens '0'–'8', then softmax → probabilities.
    Enables proper weighted ensemble without running full generation.
    """
    label_ids = get_label_token_ids(tokenizer)
    FastLanguageModel.for_inference(model)
    all_probs = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc   = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_LENGTH,
        ).to("cuda")

        with torch.no_grad():
            logits_full = model(**enc).logits   # (B, seq, vocab)

        last_pos = enc["attention_mask"].sum(dim=1) - 1  # last real token per seq

        for j, pos in enumerate(last_pos):
            l9    = torch.stack([logits_full[j, pos, tid] for tid in label_ids])
            probs = torch.softmax(l9, dim=0).cpu().numpy()
            all_probs.append(probs)

    return np.array(all_probs, dtype=np.float32)   # (N, 9)


# ── 6. Metrics ───────────────────────────────────────────────────────────────
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
        ("",       "Precision", "Recall", "F1"),
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
    print(f"  Report → {path}")

    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
    ConfusionMatrixDisplay(cm, display_labels=ALL_LABELS).plot(ax=ax, cmap="Blues")
    ax.set_title(stem)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"{stem}_cm.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {fig_path}")


def append_metrics_csv(row: dict):
    """Appends one metrics row to the shared CSV (creates header on first write)."""
    csv_path = OUTPUT_DIR / f"metrics_{MODEL_KEY}.csv"
    df_row   = pd.DataFrame([row])
    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)
    print(f"  Metrics CSV → {csv_path}")


# ── 7. Persistence helpers ───────────────────────────────────────────────────
def _probs_dir(seed, fold):
    d = OUTPUT_DIR / f"seed_{seed}" / f"fold_{fold}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_probs(arr, seed, fold, split):
    np.save(_probs_dir(seed, fold) / f"{split}_probs.npy", arr)

def try_load_probs(seed, fold, split):
    p = _probs_dir(seed, fold) / f"{split}_probs.npy"
    return np.load(p) if p.exists() else None

def save_val_indices(idx, seed, fold):
    np.save(_probs_dir(seed, fold) / "val_indices.npy", idx)

def load_val_indices(seed, fold):
    return np.load(_probs_dir(seed, fold) / "val_indices.npy")


# ── 8. Single-fold training ──────────────────────────────────────────────────
def train_fold(seed, fold, tr_df, val_df, test_df):
    """
    Trains on tr_df (+ aug), saves (val_probs, test_probs) to Drive.
    Re-uses cached .npy if already present → safe to re-run after disconnect.
    Returns (val_probs, test_probs) shaped (N_val, 9) and (N_test, 9).
    """

    cached_val  = try_load_probs(seed, fold, "val")
    cached_test = try_load_probs(seed, fold, "test")
    if cached_val is not None and cached_test is not None:
        print(f"  [seed={seed} fold={fold}] Using cached predictions — skipping training.")
        return cached_val, cached_test

    # Hesitation aug: fold-level, leak-free (generated from this fold's train only)
    aug_hes = augment_fold(tr_df)

    # Back-translation: pre-saved parquet, filtered to this fold's dialogue_ids
    aug_bt   = load_aug_data()
    parts    = [tr_df]
    if len(aug_hes):
        parts.append(aug_hes)
    if aug_bt is not None:
        fold_ids = set(tr_df["dialogue_id"])
        aug_bt   = aug_bt[aug_bt["dialogue_id"].str.replace("_bac", "").isin(fold_ids)]
        parts.append(aug_bt)
    tr_df = pd.concat(parts, ignore_index=True)

    print(f"\n  [seed={seed} fold={fold}]  train={len(tr_df)}  val={len(val_df)}  test={len(test_df)}")

    model, tokenizer = load_model_and_tokenizer(seed)

    train_ds = df_to_hf(tr_df,  tokenizer, is_training=True)
    val_ds   = df_to_hf(val_df,  tokenizer, is_training=False)
    test_ds  = df_to_hf(test_df, tokenizer, is_training=False)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ckpt_dir = OUTPUT_DIR / f"seed_{seed}" / f"fold_{fold}"

    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = train_ds,
        args = SFTConfig(
            dataset_text_field          = "text",
            dataset_num_proc            = 2,
            packing                     = False,
            output_dir                  = str(ckpt_dir),
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            num_train_epochs            = EPOCHS,
            learning_rate               = LR,
            weight_decay                = WEIGHT_DECAY,
            fp16                        = not use_bf16,
            bf16                        = use_bf16,
            warmup_steps                = 20,
            logging_steps               = 10,
            save_strategy               = "epoch",
            save_total_limit            = 1,
            optim                       = "adamw_8bit",
            lr_scheduler_type           = "cosine",
            seed                        = seed,
            report_to                   = "none",
        ),
    )
    trainer.train()
    del trainer, model, tokenizer
    del train_ds , val_ds , test_ds
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    val_probs  = predict_label_probs(model, tokenizer, val_ds["text"])
    test_probs = predict_label_probs(model, tokenizer, test_ds["text"])

    save_probs(val_probs,  seed, fold, "val")
    save_probs(test_probs, seed, fold, "test")


    return val_probs, test_probs


# ── 9. Main CV loop ──────────────────────────────────────────────────────────
def main():
    train_df, test_df, group_col = load_data()

    groups = (
        train_df[group_col].values
        if group_col else np.arange(len(train_df))
    )
    sgkf   = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(sgkf.split(train_df, train_df["label"], groups=groups))

    oof_probs_by_seed  = {}
    test_probs_by_seed = {}

    for seed in SEEDS:
        print(f"\n{'='*55}\nSEED {seed}\n{'='*55}")
        oof_probs       = np.zeros((len(train_df), NUM_CLASSES), dtype=np.float32)
        fold_test_probs = []

        for fold, (tr_idx, val_idx) in enumerate(splits):
            tr_df  = train_df.iloc[tr_idx].reset_index(drop=True)
            val_df = train_df.iloc[val_idx].reset_index(drop=True)

            save_val_indices(val_idx, seed, fold)
            val_probs, test_probs = train_fold(seed, fold, tr_df, val_df, test_df)

            oof_probs[val_idx] = val_probs
            fold_test_probs.append(test_probs)

            m = compute_metrics(val_df["label"].values, val_probs.argmax(axis=1))
            print(f"  fold {fold} val →  f1_all={m['f1_all']:.4f}  f1_pos={m['f1_pos']:.4f}")
            append_metrics_csv({"split": "val_fold", "seed": seed, "fold": fold,
                                "weights": str(SEED_WEIGHTS), **m})

        oof_probs_by_seed[seed]  = oof_probs
        test_probs_by_seed[seed] = np.mean(fold_test_probs, axis=0)

        m = compute_metrics(train_df["label"].values, oof_probs.argmax(axis=1))
        print_metrics(m, f"OOF — seed {seed}")
        append_metrics_csv({"split": "oof_seed", "seed": seed, "fold": "all",
                            "weights": str(SEED_WEIGHTS), **m})

    # ── 10. Seeded ensemble ───────────────────────────────────────────────────
    assert len(SEEDS) == len(SEED_WEIGHTS) and abs(sum(SEED_WEIGHTS) - 1) < 1e-6

    oof_ensemble  = sum(w * oof_probs_by_seed[s]  for w, s in zip(SEED_WEIGHTS, SEEDS))
    test_ensemble = sum(w * test_probs_by_seed[s] for w, s in zip(SEED_WEIGHTS, SEEDS))

    oof_preds  = oof_ensemble.argmax(axis=1)
    test_preds = test_ensemble.argmax(axis=1)

    print_metrics(compute_metrics(train_df["label"].values, oof_preds),
                  f"OOF Ensemble  [{MODEL_KEY}]  weights={SEED_WEIGHTS}")
    print_metrics(compute_metrics(test_df["label"].values,  test_preds),
                  f"Test Ensemble [{MODEL_KEY}]  weights={SEED_WEIGHTS}")

    save_report(train_df["label"].values, oof_preds,  f"oof_{MODEL_KEY}")
    save_report(test_df["label"].values,  test_preds, f"test_{MODEL_KEY}")

    append_metrics_csv({"split": "oof_ensemble",  "seed": "ensemble", "fold": "all",
                        "weights": str(SEED_WEIGHTS),
                        **compute_metrics(train_df["label"].values, oof_preds)})
    append_metrics_csv({"split": "test_ensemble", "seed": "ensemble", "fold": "all",
                        "weights": str(SEED_WEIGHTS),
                        **compute_metrics(test_df["label"].values, test_preds)})

    np.save(OUTPUT_DIR / "oof_ensemble_probs.npy",  oof_ensemble)
    np.save(OUTPUT_DIR / "test_ensemble_probs.npy", test_ensemble)
    print(f"\nSaved probs → {OUTPUT_DIR}")


# ============================================================================
# INFERENCE-ONLY SECTION
# Run this block alone (after training) to re-evaluate with different weights
# or to load from a checkpoint path directly.
# ============================================================================

def run_inference_only(seed_weights=None, from_checkpoints=False):
    """
    Assembles ensemble from saved .npy probs (fast) or re-runs model inference
    from saved checkpoints (when from_checkpoints=True).
    Prints and saves full metrics for OOF + test.
    """
    if seed_weights is None:
        seed_weights = SEED_WEIGHTS

    train_df, test_df, group_col = load_data()

    oof_acc  = {}
    test_acc = {}

    for seed in SEEDS:
        if from_checkpoints:
            # ── reload from saved LoRA checkpoints ───────────────────────────
            groups = (
                train_df[group_col].values
                if group_col else np.arange(len(train_df))
            )
            sgkf_   = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            splits_ = list(sgkf_.split(train_df, train_df["label"], groups=groups))

            oof_probs = np.zeros((len(train_df), NUM_CLASSES), dtype=np.float32)
            fold_test = []

            for fold, (_, val_idx) in enumerate(splits_):
                ckpt_dir = OUTPUT_DIR / f"seed_{seed}" / f"fold_{fold}"
                checkpoints = sorted(ckpt_dir.glob("checkpoint-*"))
                if not checkpoints:
                    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
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
                val_ds  = df_to_hf(val_df,  tokenizer, is_training=False)
                test_ds = df_to_hf(test_df, tokenizer, is_training=False)

                val_probs  = predict_label_probs(model, tokenizer, val_ds["text"])
                test_probs = predict_label_probs(model, tokenizer, test_ds["text"])

                oof_probs[val_idx] = val_probs
                fold_test.append(test_probs)

                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()

            oof_acc[seed]  = oof_probs
            test_acc[seed] = np.mean(fold_test, axis=0)

        else:
            # ── load cached .npy (fast path) ─────────────────────────────────
            oof_probs = np.zeros((len(train_df), NUM_CLASSES), dtype=np.float32)
            fold_test = []
            for fold in range(N_FOLDS):
                val_idx   = load_val_indices(seed, fold)
                val_probs = try_load_probs(seed, fold, "val")
                t_probs   = try_load_probs(seed, fold, "test")
                if val_probs is None or t_probs is None:
                    raise FileNotFoundError(
                        f"Missing probs for seed={seed} fold={fold}. "
                        "Run training first or use from_checkpoints=True."
                    )
                oof_probs[val_idx] = val_probs
                fold_test.append(t_probs)
            oof_acc[seed]  = oof_probs
            test_acc[seed] = np.mean(fold_test, axis=0)

    oof_ens  = sum(w * oof_acc[s]  for w, s in zip(seed_weights, SEEDS))
    test_ens = sum(w * test_acc[s] for w, s in zip(seed_weights, SEEDS))

    tag = f"w{'_'.join(str(w) for w in seed_weights)}"
    print_metrics(compute_metrics(train_df["label"].values, oof_ens.argmax(axis=1)),
                  f"OOF  [{MODEL_KEY}] {tag}")
    print_metrics(compute_metrics(test_df["label"].values,  test_ens.argmax(axis=1)),
                  f"Test [{MODEL_KEY}] {tag}")

    save_report(train_df["label"].values, oof_ens.argmax(axis=1),
                f"oof_{MODEL_KEY}_{tag}")
    save_report(test_df["label"].values,  test_ens.argmax(axis=1),
                f"test_{MODEL_KEY}_{tag}")


# ── Example: try a different weight split without retraining ─────────────────
# run_inference_only(seed_weights=[0.3, 0.7])
# run_inference_only(from_checkpoints=True)

if __name__ == "__main__":
    main()
