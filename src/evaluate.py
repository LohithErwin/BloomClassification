import os, sys, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import load_and_preprocess, split_data
from src.train import BloomDataset

MODEL_SAVE = "models/distilbert_blooms"
BATCH_SIZE = 32
MAX_LENGTH = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOOM_NAMES = {"BT1":"Remembering","BT2":"Understanding","BT3":"Applying",
               "BT4":"Analyzing","BT5":"Evaluating","BT6":"Creating"}

def evaluate():
    df, le    = load_and_preprocess()
    _, _, test_df = split_data(df)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_SAVE)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_SAVE).to(DEVICE)
    label_classes = np.load(os.path.join(MODEL_SAVE, "label_classes.npy"), allow_pickle=True)

    test_ds     = BloomDataset(test_df["cleaned_text"], test_df["label"], tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            out   = model(input_ids=batch["input_ids"].to(DEVICE),
                          attention_mask=batch["attention_mask"].to(DEVICE))
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(batch["labels"].numpy())

    target_names = [f"{c} ({BLOOM_NAMES[c]})" for c in label_classes]
    print(classification_report(y_true, y_pred, target_names=target_names))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_classes, yticklabels=label_classes)
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")

if __name__ == "__main__":
    evaluate()