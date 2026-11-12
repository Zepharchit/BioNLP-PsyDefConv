# ============================================================================
# quick_infer.py  —  temp inference: load saved adapter, run on test set
#
# Usage:
#   python quick_infer.py --model lora --single        # single-train adapter
#   python quick_infer.py --model lora --fold 0        # fold-0 adapter (CV)
#   python quick_infer.py --model qlora --single
# ============================================================================

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",  choices=["lora", "qlora"], default="lora")
parser.add_argument("--single", action="store_true")
parser.add_argument("--fold",   type=int, default=0,
                    help="Which fold adapter to load (ignored with --single)")
args = parser.parse_args()

STAGE2_DIRS = {
    "lora":  Path("/root/output/stage2_lora"),
    "qlora": Path("/root/output/stage2_qlora"),
}
OUTPUT_DIR = STAGE2_DIRS[args.model]
subdir     = "single" if args.single else f"fold_{args.fold}"
ADAPTER    = OUTPUT_DIR / subdir / "adapter"

if not ADAPTER.exists():
    raise FileNotFoundError(f"No adapter at {ADAPTER}")

print(f"Loading adapter from {ADAPTER}")

MAX_LENGTH  = 1024
NUM_CLASSES = 8

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
    turns = turns[-20:]
    context = "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in turns)

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


# ── Load tokenizer + model ────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
if tokenizer.pad_token is None:
    cand = "<|finetune_right_pad_id|>"
    tokenizer.pad_token = cand if cand in tokenizer.get_vocab() else tokenizer.eos_token
tokenizer.padding_side = "right"

base_name = json.loads((ADAPTER / "adapter_config.json").read_text())["base_model_name_or_path"]
print(f"Base model: {base_name}")

use_bf16      = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

base = AutoModelForSequenceClassification.from_pretrained(
    base_name,
    num_labels=NUM_CLASSES,
    torch_dtype=compute_dtype,
    device_map={"": 0},
    attn_implementation="sdpa",
)
base.config.pad_token_id = tokenizer.pad_token_id

model = PeftModel.from_pretrained(base, str(ADAPTER))
model.eval()
print("Model loaded.")

# ── Load test data ────────────────────────────────────────────────────────────
print("Loading test set...")
ds      = load_dataset("AIMH/PsyDefConv")
test_df = ds["test"].to_pandas()
test_df["label"] = test_df["label"].astype(int)

# Keep only pos samples (gate would filter 0s; here we score all for reference)
pos_mask = test_df["label"] != 0
print(f"Test samples: {len(test_df)}  pos (1-8): {pos_mask.sum()}")

# ── Inference ─────────────────────────────────────────────────────────────────
all_logits = []
BATCH = 8
rows = test_df[pos_mask].reset_index(drop=True)

with torch.no_grad():
    for start in range(0, len(rows), BATCH):
        batch_rows = rows.iloc[start : start + BATCH]
        prompts = [
            build_prompt(str(r["current_text"]), r.get("dialogue"), tokenizer)
            for _, r in batch_rows.iterrows()
        ]
        enc = tokenizer(
            prompts, truncation=True, max_length=MAX_LENGTH,
            padding=True, return_tensors="pt",
        ).to(model.device)
        out = model(**enc)
        all_logits.append(out.logits.float().cpu().numpy())
        if (start // BATCH) % 10 == 0:
            print(f"  {start}/{len(rows)}", end="\r")

print(f"\nDone — {len(rows)} samples")

logits = np.concatenate(all_logits, axis=0)     # (n_pos, 8)
preds  = logits.argmax(-1) + 1                  # remap 0-7 → 1-8
y_true = rows["label"].values                   # original labels 1-8

# ── Report ────────────────────────────────────────────────────────────────────
print("\n" + classification_report(
    y_true, preds,
    labels=list(range(1, 9)),
    target_names=[f"{i}: {DMRS_LABELS[i]}" for i in range(1, 9)],
    digits=4, zero_division=0,
))

# Save logits for optional further use
out_path = OUTPUT_DIR / subdir / "quick_test_logits.npy"
np.save(out_path, logits)
print(f"Logits saved → {out_path}")
print("\nBaseline to beat (Na et al., Ministral-8B): pos=0.3148  all=0.3745")
