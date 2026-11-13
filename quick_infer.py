
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# ── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["lora", "qlora"], default="lora")
parser.add_argument("--single", action="store_true")
parser.add_argument("--fold", type=int, default=0)
args = parser.parse_args()

OUTPUT_DIR = Path(f"/workspace/output/stage2_qlora")
subdir = "single" if args.single else f"fold_{args.fold}"
SAVE_DIR = OUTPUT_DIR / subdir
ADAPTER = SAVE_DIR / "adapter"

if not ADAPTER.exists():
    raise FileNotFoundError(f"Adapter not found: {ADAPTER}")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────

MAX_LENGTH = 1024
NUM_CLASSES = 8
BATCH = 8

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

LABELS = list(range(1, 9))
TARGET_NAMES = [f"{i}: {DMRS_LABELS[i]}" for i in LABELS]

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

# ── Prompt ───────────────────────────────────────────────────────────────────

def parse_dialogue(d):
    if d is None:
        return []
    if isinstance(d, str):
        d = json.loads(d)
    return list(d)


def build_prompt(text, dialogue, tokenizer):
    turns = parse_dialogue(dialogue)

    if turns and str(turns[-1]["text"]).strip() == str(text).strip():
        turns = turns[:-1]

    context = "\n".join(
        f"{t['speaker'].upper()}: {t['text']}"
        for t in turns[-20:]
    )

    head = PAPER_INTRO
    target = f"\n- Target utterance:\n{text}\n"
    tail = PAPER_LABELS + PAPER_OUTPUT
    ctx_block = ""

    if context:
        fixed = len(tokenizer(
            head + target + tail,
            add_special_tokens=False,
        )["input_ids"])

        budget = MAX_LENGTH - fixed - 48

        if budget > 0:
            ids = tokenizer(
                context,
                add_special_tokens=False,
            )["input_ids"][-budget:]

            ctx_block = "\n- Dialogue context:\n" + tokenizer.decode(ids) + "\n"

    prompt = head + ctx_block + target + tail

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


# ── Load model ───────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

if tokenizer.pad_token is None:
    candidate = "<|finetune_right_pad_id|>"
    tokenizer.pad_token = (
        candidate if candidate in tokenizer.get_vocab()
        else tokenizer.eos_token
    )

tokenizer.padding_side = "right"

base_name = json.loads(
    (ADAPTER / "adapter_config.json").read_text()
)["base_model_name_or_path"]

dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

base = AutoModelForSequenceClassification.from_pretrained(
    base_name,
    num_labels=NUM_CLASSES,
    torch_dtype=dtype,
    device_map={"": 0},
    attn_implementation="sdpa",
)

base.config.pad_token_id = tokenizer.pad_token_id
model = PeftModel.from_pretrained(base, str(ADAPTER))
model.eval()

print(f"Loaded {args.model.upper()} | {subdir}")

# ── Load data ────────────────────────────────────────────────────────────────

ds = load_dataset("AIMH/PsyDefConv")
test_df = ds["test"].to_pandas()
test_df["label"] = test_df["label"].astype(int)

rows = test_df[test_df["label"] != 0].reset_index(drop=True)

if rows.empty:
    raise ValueError("No positive test samples found.")

# ── Inference ────────────────────────────────────────────────────────────────

all_logits = []

with torch.inference_mode():
    for start in range(0, len(rows), BATCH):
        batch = rows.iloc[start:start + BATCH]

        prompts = [
            build_prompt(str(r["current_text"]), r.get("dialogue"), tokenizer)
            for _, r in batch.iterrows()
        ]

        enc = tokenizer(
            prompts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        logits = model(**enc).logits.float().cpu().numpy()
        all_logits.append(logits)

        print(f"\rInference: {min(start + BATCH, len(rows))}/{len(rows)}",
              end="", flush=True)

print()

logits = np.concatenate(all_logits, axis=0)
preds = logits.argmax(axis=-1) + 1
y_true = rows["label"].values

# ── Metrics ──────────────────────────────────────────────────────────────────

report = classification_report(
    y_true,
    preds,
    labels=LABELS,
    target_names=TARGET_NAMES,
    output_dict=True,
    zero_division=0,
)

print(classification_report(
    y_true,
    preds,
    labels=LABELS,
    target_names=TARGET_NAMES,
    digits=4,
    zero_division=0,
))

f1_scores = {
    "per_class": {
        str(i): {
            "name": DMRS_LABELS[i],
            "f1": float(f1_score(y_true, preds, labels=[i],
                                 average="macro", zero_division=0)),
            "precision": float(report[TARGET_NAMES[i-1]]["precision"]),
            "recall": float(report[TARGET_NAMES[i-1]]["recall"]),
            "support": int(report[TARGET_NAMES[i-1]]["support"]),
        }
        for i in LABELS
    },
    "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
    "micro_f1": float(f1_score(y_true, preds, average="micro", zero_division=0)),
    "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
    "accuracy": float(report["accuracy"]),
    "total_samples": int(len(y_true)),
}

with open(SAVE_DIR / "f1_scores.json", "w") as f:
    json.dump(f1_scores, f, indent=2)

with open(SAVE_DIR / "classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

# ── Confusion matrix ─────────────────────────────────────────────────────────

cm = confusion_matrix(y_true, preds, labels=LABELS)

np.save(SAVE_DIR / "confusion_matrix.npy", cm)

pd.DataFrame(
    cm,
    index=TARGET_NAMES,
    columns=TARGET_NAMES,
).to_csv(SAVE_DIR / "confusion_matrix.csv")

fig, ax = plt.subplots(figsize=(11, 9))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABELS,
).plot(ax=ax, cmap="Blues", values_format="d")

ax.set_title(f"DMRS Confusion Matrix - {args.model.upper()} ({subdir})")
plt.tight_layout()
plt.savefig(SAVE_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ── Save logits ──────────────────────────────────────────────────────────────

np.save(SAVE_DIR / "quick_test_logits.npy", logits)

print(
    f"\nMacro F1: {f1_scores['macro_f1']:.4f}"
    f" | Weighted F1: {f1_scores['weighted_f1']:.4f}"
    f" | Accuracy: {f1_scores['accuracy']:.4f}"
)

print(f"Results: {SAVE_DIR}")
