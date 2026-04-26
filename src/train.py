import os, sys, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import load_and_preprocess, split_data, clean_text, MODEL_NAME

EPOCHS        = 15
BATCH_SIZE    = 32
LEARNING_RATE = 2e-5
MAX_LENGTH    = 128
MODEL_SAVE    = "models/distilbert_blooms/test"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BloomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(list(texts), truncation=True,
                                   padding="max_length", max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels":         torch.tensor(self.labels[idx]),
        }


def main():
    print("STEP 1 — Loading data …")
    df, le = load_and_preprocess()
    train_df, val_df, _ = split_data(df)
    num_labels = df["label"].nunique()

    print("STEP 2 — Tokenizing …")
    tokenizer  = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_ds   = BloomDataset(train_df["cleaned_text"], train_df["label"], tokenizer, MAX_LENGTH)
    val_ds     = BloomDataset(val_df["cleaned_text"],   val_df["label"],   tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    print("STEP 3 — Building model …")
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out  = model(input_ids=batch["input_ids"].to(DEVICE),
                         attention_mask=batch["attention_mask"].to(DEVICE),
                         labels=batch["labels"].to(DEVICE))
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                out   = model(input_ids=batch["input_ids"].to(DEVICE),
                              attention_mask=batch["attention_mask"].to(DEVICE))
                preds = out.logits.argmax(dim=-1).cpu()
                correct += (preds == batch["labels"]).sum().item()
                total   += len(batch["labels"])
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODEL_SAVE, exist_ok=True)
            model.save_pretrained(MODEL_SAVE)
            tokenizer.save_pretrained(MODEL_SAVE)
            np.save(os.path.join(MODEL_SAVE, "label_classes.npy"), le.classes_)
            print(f"Best model saved (val_acc={val_acc*100:.2f}%)")

    print(f"\nBest Validation Accuracy: {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    main()