import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME  = "unsloth/Llama-3.1-8B-Instruct"
MAX_LENGTH  = 1024

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


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    cand = "<|finetune_right_pad_id|>"
    tokenizer.pad_token = cand if cand in tokenizer.get_vocab() else tokenizer.eos_token

print("Loading dataset...")
ds       = load_dataset("AIMH/PsyDefConv")
train_df = ds["train"].to_pandas()
test_df  = ds["test"].to_pandas()

for split_name, df in [("train", train_df), ("test", test_df)]:
    lengths = []
    for _, row in df.iterrows():
        context = context_to_text(row.get("dialogue"), str(row["current_text"]))
        prompt  = build_prompt(str(row["current_text"]), context, tokenizer)
        ids     = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)["input_ids"]
        lengths.append(len(ids))
    lengths = np.array(lengths)
    print(f"\n{split_name} ({len(df)} samples):")
    print(f"  mean={lengths.mean():.0f}  median={int(np.median(lengths))}  "
          f"p95={int(np.percentile(lengths,95))}  max={lengths.max()}  min={lengths.min()}")
    print(f"  % hitting MAX_LENGTH ({MAX_LENGTH}): {(lengths==MAX_LENGTH).mean()*100:.1f}%")
