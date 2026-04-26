import os, sys, numpy as np, torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import clean_text

MODEL_SAVE = "models/distilbert_blooms"
MAX_LENGTH = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOOM_DETAILS = {
    "BT1": ("Remembering",   "Recall or recognize basic facts and concepts."),
    "BT2": ("Understanding", "Comprehend and explain ideas or concepts."),
    "BT3": ("Applying",      "Use knowledge in new situations or solve problems."),
    "BT4": ("Analyzing",     "Break down information to explore relationships."),
    "BT5": ("Evaluating",    "Make judgments based on criteria or standards."),
    "BT6": ("Creating",      "Generate new ideas, products, or approaches."),
}


def load_model_and_tokenizer():
    tokenizer     = DistilBertTokenizerFast.from_pretrained(MODEL_SAVE)
    model         = DistilBertForSequenceClassification.from_pretrained(MODEL_SAVE).to(DEVICE)
    label_classes = np.load(os.path.join(MODEL_SAVE, "label_classes.npy"), allow_pickle=True)
    model.eval()
    return model, tokenizer, label_classes


def predict(text, model, tokenizer, label_classes):
    cleaned  = clean_text(text)
    encoding = tokenizer(cleaned, truncation=True, padding="max_length",
                         max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=encoding["input_ids"].to(DEVICE),
                       attention_mask=encoding["attention_mask"].to(DEVICE)).logits
    probs      = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_idx   = int(np.argmax(probs))
    pred_label = label_classes[pred_idx]
    return {
        "label":      pred_label,
        "level_name": BLOOM_DETAILS[pred_label][0],
        "desc":       BLOOM_DETAILS[pred_label][1],
        "confidence": float(probs[pred_idx]),
        "all_probs":  {label_classes[i]: float(probs[i]) for i in range(len(probs))},
    }


def print_result(text, result):
    print("\n" + "=" * 60)
    print(f"  Input     : {text}")
    print(f"  Level     : {result['label']} — {result['level_name']}")
    print(f"  Detail    : {result['desc']}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    print("\n  All probabilities:")
    for lbl, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        print(f"    {lbl}: {prob*100:5.1f}% ")
    print("=" * 60)


if __name__ == "__main__":
    print("Loading model …")
    model, tokenizer, label_classes = load_model_and_tokenizer()
    print("Model ready! Type your question and press Enter. (Ctrl+C to quit)\n")

    while True:
        text = input("Enter question: ").strip()
        if not text:
            print("Please enter a valid question.\n")
            continue
        result = predict(text, model, tokenizer, label_classes)
        print_result(text, result)
        print()