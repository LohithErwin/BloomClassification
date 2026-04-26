import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast

DATA_PATH   = "data/blooms_taxonomy_dataset.csv"
MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 128
RANDOM_SEED = 42

LABEL_MAP = {
    "BT1": "Remembering",
    "BT2": "Understanding",
    "BT3": "Applying",
    "BT4": "Analyzing",
    "BT5": "Evaluating",
    "BT6": "Creating",
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s']", " ", text)   
    text = re.sub(r"\s+", " ", text).strip() 
    return text


def load_and_preprocess(data_path: str = DATA_PATH):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()          
    df = df[["Questions", "Category"]].dropna()  

    df["cleaned_text"] = df["Questions"].apply(clean_text)

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Category"])

    print(f"Dataset loaded: {len(df)} samples, {df['label'].nunique()} classes")
    print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    return df, le


def split_data(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15):
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=RANDOM_SEED
    )
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val["label"], random_state=RANDOM_SEED
    )
    print(f"Split → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test

