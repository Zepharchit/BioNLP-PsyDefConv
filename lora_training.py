# ============================================================================
# lora_training.py
# Stage 2: fine-tune LLaMA 3.1 8B classifier on labels 1-8 only (3-fold CV).
# LoRA only — no quantization. Loads model in bf16/fp16.
# Class 0 is handled by the Stage 1 gate (stage1_binary_clf.py).
# Labels internally remapped: 1-8 → 0-7 for the 8-class model head.
# Saves per-fold test_logits.npy (shape: n_test × 8) → consumed by final_inference.py
# ============================================================================

import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    EarlyStoppingCallback, Trainer, TrainingArguments, set_seed,
)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"

OUTPUT_DIR = Path("/root/workspace/stage2_lora")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
N_FOLDS     = 3
NUM_CLASSES = 8           # 8 output neurons; labels 1-8 remapped to 0-7
ALL         = list(range(NUM_CLASSES))   # [0..7]

MAX_LENGTH      = 1024
EPOCHS          = 10
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 0.01
MAX_GRAD_NORM   = 0.3
BATCH_SIZE      = 8
GRAD_ACCUM      = 2
EVAL_BATCH_SIZE = 16

LORA_R        = 32
LORA_ALPHA    = 64
LORA_DROPOUT  = 0.05

LABEL_SMOOTHING = 0.05

MINORITY_LABELS = [1, 2, 3, 4, 5, 8]
TARGET_COUNT    = 200

# Original DMRS labels 1-8 (for display only)
DMRS_LABELS = {
    1: "Action Defense Level",
    2: "Major Image-distorting Defense Level",
    3: "Disavowal Defense Level",
    4: "Minor Image-distorting Defense Level",
    5: "Neurotic Defense Level",
    6: "Obsessional Defense Level",
    7: "Highly Adaptive Defense Level",
    8: "Need More Information",
}

PAPER_INTRO = (
    "You are a Defense Mechanism Rating Scale (DMRS) specialist. "
    "Examine the dialog & utterance carefully and select the single most "
    "appropriate defense tier. When multiple defenses seem plausible, choose "
    "the tier with the strongest supporting evidence.\n"
)

PAPER_LABELS = (
    "Available labels (DMRS tiers):\n"
    "1 = Action Defense Level (Acting Out / Help-Rejecting Complaining / Passive Aggression)\n"
    "2 = Major Image-distorting Defense Level (Splitting / Projective Identification)\n"
    "3 = Disavowal Defense Level (Denial / Projection / Rationalization / Autistic Fantasy)\n"
    "4 = Minor Image-distorting Defense Level (Devaluation / Idealization / Omnipotence)\n"
    "5 = Neurotic Defense Level (Displacement / Dissociation / Reaction Formation / Repression)\n"
    "6 = Obsessional Defense Level (Intellectualization / Isolation of Affects / Undoing)\n"
    "7 = Highly Adaptive Defense Level (Affiliation / Altruism / Anticipation / Humor / "
    "Self-Assertion / Self-Observation / Sublimation / Suppression)\n"
    "8 = Need More Information\n"
)

PAPER_OUTPUT = "Return exactly one line:\nlabel: <1-8>\n"


# ── Data utilities ────────────────────────────────────────────────────────────
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
    turns = turns[-20:]
    return "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in turns)


def build_prompt(text, context, tokenizer):
    head   = PAPER_INTRO
    target = f"\n- Target utterance:\n{text}\n"
    tail   = PAPER_LABELS + PAPER_OUTPUT

    ctx_block = ""
    if context:
        fixed  = len(tokenizer(head + target + tail, add_special_tokens=False)["input_ids"])
        budget = MAX_LENGTH - fixed - 48
        if budget > 0:
            ids = tokenizer(context, add_special_tokens=False)["input_ids"][-budget:]
            ctx_block = "\n- Dialogue context:\n" + tokenizer.decode(ids) + "\n"

    user_prompt = head + ctx_block + target + tail
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    return user_prompt


def tokenize_dataframe(df, tokenizer, remap_labels=True):
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    for _, row in df.iterrows():
        context = context_to_text(row.get("dialogue"), str(row["current_text"]))
        prompt  = build_prompt(str(row["current_text"]), context, tokenizer)
        enc = tokenizer(
            prompt, truncation=True, max_length=MAX_LENGTH,
            add_special_tokens=not bool(getattr(tokenizer, "chat_template", None)),
        )
        rows["input_ids"].append(enc["input_ids"])
        rows["attention_mask"].append(enc["attention_mask"])
        label = int(row["label"])
        rows["labels"].append(label - 1 if remap_labels else label)
    return Dataset.from_dict(rows)


def load_data():
    ds       = load_dataset("AIMH/PsyDefConv")
    train_df = ds["train"].to_pandas()
    test_df  = ds["test"].to_pandas()
    for df in (train_df, test_df):
        df["label"] = df["label"].astype(int)
    group_col = "dialogue_id" if "dialogue_id" in train_df.columns else None
    print(f"Train: {len(train_df)}  Test: {len(test_df)}  Group: {group_col}")
    return train_df, test_df, group_col


def get_folds(df, group_col):
    if group_col:
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        return list(sgkf.split(df, df["label"], groups=df[group_col]))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(df, df["label"]))


def augment_fold(tr_df):
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
        print(f"    augment_fold label {label}: +{n} rows")

    if not rows:
        return pd.DataFrame(columns=tr_df.columns)

    aug = pd.concat(rows).reset_index(drop=True)
    aug["dialogue_id"] = aug["dialogue_id"] + "_hes"
    return aug


# ── Model ────────────────────────────────────────────────────────────────────
def inverse_sqrt_weights(labels_07):
    counts = np.bincount(np.asarray(labels_07, dtype=int), minlength=NUM_CLASSES)
    w = 1.0 / np.sqrt(np.maximum(counts, 1).astype(np.float64))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32), counts


def load_model():
    set_seed(SEED)
    use_bf16      = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        cand = "<|finetune_right_pad_id|>"
        tokenizer.pad_token = cand if cand in tokenizer.get_vocab() else tokenizer.eos_token
    tokenizer.padding_side = "right"

    id2label = {i: DMRS_LABELS[i + 1] for i in ALL}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        problem_type="single_label_classification",
        torch_dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache    = False
    model.config.id2label     = id2label
    model.config.label2id     = label2id

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ── Trainer ───────────────────────────────────────────────────────────────────
class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        feats = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = self.tokenizer.pad(feats, padding=True, return_tensors="pt")
        if "labels" in features[0]:
            batch["labels"] = torch.tensor(
                [f["labels"] for f in features], dtype=torch.long
            )
        return batch


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, smoothing=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.smoothing     = smoothing

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits.float()
        loss = F.cross_entropy(
            logits, labels,
            weight=self.class_weights.to(logits.device),
            label_smoothing=self.smoothing,
        )
        return (loss, outputs) if return_outputs else loss


# ── Metrics (all labels 0-7, representing original 1-8) ──────────────────────
def metrics_from(y_true, y_pred):
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ALL, average="macro", zero_division=0,
    )
    out["macro_p"]   = p
    out["macro_r"]   = r
    out["macro_f1"]  = f
    return out


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    return metrics_from(eval_pred.label_ids, np.argmax(logits, axis=-1))


def make_training_args(out_dir, fold_idx):
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return TrainingArguments(
        output_dir                    = str(out_dir),
        learning_rate                 = LEARNING_RATE,
        per_device_train_batch_size   = BATCH_SIZE,
        per_device_eval_batch_size    = EVAL_BATCH_SIZE,
        gradient_accumulation_steps   = GRAD_ACCUM,
        num_train_epochs              = EPOCHS,
        weight_decay                  = WEIGHT_DECAY,
        lr_scheduler_type             = "cosine",
        warmup_steps                  = 100,
        max_grad_norm                 = MAX_GRAD_NORM,
        eval_strategy                 = "epoch",
        save_strategy                 = "epoch",
        save_total_limit              = 1,
        load_best_model_at_end        = True,
        metric_for_best_model         = "eval_macro_f1",
        greater_is_better             = True,
        logging_steps                 = 100,
        logging_first_step            = True,
        report_to                     = "tensorboard",
        bf16                          = use_bf16,
        fp16                          = not use_bf16,
        optim                         = "paged_adamw_8bit",
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        remove_unused_columns         = False,
        seed                          = SEED,
        data_seed                     = SEED + fold_idx,
    )


def get_logits(trainer, dataset):
    ds = dataset.remove_columns("labels") if "labels" in dataset.column_names else dataset
    out    = trainer.predict(ds)
    logits = out.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    return np.asarray(logits)


def save_eval(tag, logits, y_true_07, out_dir):
    np.save(out_dir / f"{tag}_logits.npy", logits)
    y_pred  = logits.argmax(-1)
    metrics = metrics_from(y_true_07, y_pred)

    report = classification_report(
        y_true_07, y_pred, labels=ALL,
        target_names=[f"{i+1}: {DMRS_LABELS[i+1]}" for i in ALL],
        digits=4, zero_division=0,
    )
    (out_dir / f"{tag}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true_07, y_pred, labels=ALL)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=[i + 1 for i in ALL]).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False,
    )
    ax.set_title(f"DMRS confusion matrix ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_confusion_matrix.png", dpi=200)
    plt.close(fig)
    return metrics, report


# ── Per-fold run ──────────────────────────────────────────────────────────────
def run_fold(fold_idx, train_df, val_df, test_df, out_subdir=None):
    out_dir = OUTPUT_DIR / (out_subdir or f"fold_{fold_idx}")
    out_dir.mkdir(parents=True, exist_ok=True)

    weights, counts = inverse_sqrt_weights(train_df["label"].values - 1)
    print(f"\n[fold {fold_idx}] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print("class counts (0-7):", counts.tolist())
    print("class weights:", [round(float(x), 3) for x in weights])

    aug = augment_fold(train_df)
    if len(aug):
        train_df = pd.concat([train_df, aug], ignore_index=True)
        print(f"After aug: train={len(train_df)}")

    model, tokenizer = load_model()
    train_ds = tokenize_dataframe(train_df, tokenizer, remap_labels=True)
    val_ds   = tokenize_dataframe(val_df,   tokenizer, remap_labels=True)
    test_ds  = tokenize_dataframe(test_df,  tokenizer, remap_labels=False)

    lens = [len(x) for x in train_ds["input_ids"]]
    print(f"token length: mean={np.mean(lens):.0f}  max={max(lens)}")

    trainer = WeightedLossTrainer(
        model            = model,
        args             = make_training_args(out_dir, fold_idx),
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        data_collator    = Collator(tokenizer),
        processing_class = tokenizer,
        compute_metrics  = compute_metrics,
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights    = weights,
        smoothing        = LABEL_SMOOTHING,
    )
    trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    val_metrics, _ = save_eval(
        "val", get_logits(trainer, val_ds), val_df["label"].values - 1, out_dir,
    )

    test_logits = get_logits(trainer, test_ds)
    np.save(out_dir / "test_logits.npy", test_logits)

    pos_mask = test_df["label"].values != 0
    test_metrics, test_report = save_eval(
        "test", test_logits[pos_mask], test_df["label"].values[pos_mask] - 1, out_dir,
    )

    print(test_report)
    print(
        f"[fold {fold_idx}] TEST macro-F1={test_metrics['macro_f1']:.4f}  "
        f"acc={test_metrics['accuracy']:.4f}"
    )

    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {"fold": fold_idx,
            **{f"val_{k}": v  for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()}}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true",
                        help="Train once on full data (80/20 split) instead of 3-fold CV")
    args = parser.parse_args()

    train_full, test_df, group_col = load_data()

    train_full = train_full[train_full["label"] != 0].reset_index(drop=True)
    print(f"After class-0 filter: train={len(train_full)}")
    print("Label distribution:\n", train_full["label"].value_counts().sort_index().to_string())

    if args.single:
        if group_col:
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
            tr_idx, va_idx = next(sgkf.split(train_full, train_full["label"],
                                              groups=train_full[group_col]))
            train_df = train_full.iloc[tr_idx].reset_index(drop=True)
            val_df   = train_full.iloc[va_idx].reset_index(drop=True)
        else:
            train_df, val_df = train_test_split(
                train_full, test_size=0.2, stratify=train_full["label"], random_state=SEED,
            )
            train_df = train_df.reset_index(drop=True)
            val_df   = val_df.reset_index(drop=True)
        print(f"\n[single] train={len(train_df)}  val={len(val_df)}")
        result = run_fold(0, train_df, val_df, test_df, out_subdir="single")
        pd.DataFrame([result]).to_csv(OUTPUT_DIR / "results_single.csv", index=False)
        print(f"\n[single] TEST macro-F1={result['test_macro_f1']:.4f}  acc={result['test_accuracy']:.4f}")
    else:
        folds   = get_folds(train_full, group_col)
        results = []
        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            train_df = train_full.iloc[tr_idx].reset_index(drop=True)
            val_df   = train_full.iloc[va_idx].reset_index(drop=True)
            result   = run_fold(fold_idx, train_df, val_df, test_df)
            results.append(result)

        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_DIR / "results_by_fold.csv", index=False)
        print("\nPer-fold results:")
        print(df[["fold", "test_macro_f1", "test_accuracy"]].to_string(index=False))

    print("\nBaseline to beat (Na et al., Ministral-8B): pos=0.3148  all=0.3745")


if __name__ == "__main__":
    main()
