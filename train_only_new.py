# ============================================================================
# train_only.py
# PsyDefConv — Unsloth LLM SFT  |  Training only
# Runs CV for all SEEDS, saves per-fold probs + checkpoints to Drive.
# Change MODEL_KEY, then run. Ensemble → inference_only.py
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

OUTPUT_DIR = Path("/workspace/psydef_unsloth") / MODEL_KEY
AUG_PATH   = Path("/workspace/augmented_samples")
AUG_FILES  = []   # back-translation disabled; hesitation runs inline per fold

N_FOLDS         = 3
SEEDS           = [42, 123]
MAX_LENGTH      = 1024
EPOCHS          = 1
LR              = 2e-4
WEIGHT_DECAY    = 0.01
BATCH_SIZE      = 8
GRAD_ACCUM      = 4
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
NUM_CLASSES     = 9
ALL_LABELS      = list(range(NUM_CLASSES))
POS_LABELS      = list(range(1, NUM_CLASSES))
MINORITY_LABELS = [1, 2, 3, 4, 5, 8]
TARGET_COUNT    = 200

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Model : {MODEL_NAME}")
print(f"Output: {OUTPUT_DIR}")

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
    """Hesitation aug on one fold's train only — never touches val/test."""
    CONTRACTIONS = {
        "I am": "I'm",        "do not": "don't",       "did not": "didn't",
        "can not": "can't",   "cannot": "can't",        "will not": "won't",
        "would not": "wouldn't", "I have": "I've",      "I would": "I'd",
        "I will": "I'll",     "they are": "they're",    "we are": "we're",
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
            modes[i % 3](t) for i, t in enumerate(samples["current_text"].tolist())
        ]
        rows.append(samples)
        print(f"    augment_fold label {label}: +{n}")
    if not rows:
        return pd.DataFrame(columns=tr_df.columns)
    aug = pd.concat(rows).reset_index(drop=True)
    aug["dialogue_id"] = aug["dialogue_id"] + "_hes"
    return aug


# ── 4. Model ──────────────────────────────────────────────────────────────────
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


# ── 5. Inference ──────────────────────────────────────────────────────────────
def get_label_token_ids(tokenizer):
    return [
        tokenizer(str(i), add_special_tokens=False)["input_ids"][0]
        for i in range(NUM_CLASSES)
    ]


def predict_label_probs(model, tokenizer, prompts, batch_size=16):
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
            logits_full = model(**enc).logits
        last_pos = enc["attention_mask"].sum(dim=1) - 1
        for j, pos in enumerate(last_pos):
            l9    = torch.stack([logits_full[j, pos, tid] for tid in label_ids])
            probs = torch.softmax(l9, dim=0).cpu().numpy()
            all_probs.append(probs)
    return np.array(all_probs, dtype=np.float32)


# ── 6. Metrics ────────────────────────────────────────────────────────────────
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


def append_metrics_csv(row: dict):
    csv_path = OUTPUT_DIR / f"metrics_{MODEL_KEY}.csv"
    df_row   = pd.DataFrame([row])
    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)
    print(f"  Metrics CSV → {csv_path}")


# ── 7. Persistence ────────────────────────────────────────────────────────────
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


# ── 8. Single-fold training ───────────────────────────────────────────────────
def train_fold(seed, fold, tr_df, val_df, test_df):
    ckpt_dir   = OUTPUT_DIR / f"seed_{seed}" / f"fold_{fold}"
    checkpoints = sorted(ckpt_dir.glob("checkpoint-*"))
    if checkpoints:
        print(f"  [seed={seed} fold={fold}] checkpoint exists — skipping.")
        return None, None

    aug_hes = augment_fold(tr_df)
    aug_bt  = load_aug_data()
    parts   = [tr_df]
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

    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = train_ds,
        args = SFTConfig(
            dataset_text_field          = "text",
            max_seq_length              = MAX_LENGTH,
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

    del trainer, model, tokenizer, train_ds, val_ds, test_ds
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  [seed={seed} fold={fold}] checkpoint saved to {ckpt_dir}")
    return None, None


# ── 9. CV loop ────────────────────────────────────────────────────────────────
train_df, test_df, group_col = load_data()

groups = (
    train_df[group_col].values
    if group_col else np.arange(len(train_df))
)
sgkf   = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
splits = list(sgkf.split(train_df, train_df["label"], groups=groups))

for seed in SEEDS:
    print(f"\n{'='*55}\nSEED {seed}\n{'='*55}")

    for fold, (tr_idx, val_idx) in enumerate(splits):
        tr_df  = train_df.iloc[tr_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)

        save_val_indices(val_idx, seed, fold)
        train_fold(seed, fold, tr_df, val_df, test_df)

print(f"\nTraining complete. Checkpoints saved to {OUTPUT_DIR}")
