
import pickle
import numpy as np
from datasets import load_dataset

MAX_LENGTH = 100

print("Loading dataset and tokenizer...")
dataset = load_dataset("fancyzhx/ag_news")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

for split_name in ["train", "test"]:
    texts = dataset[split_name]["text"]

    sequences = tokenizer.texts_to_sequences(texts)
    lengths = np.array([len(seq) for seq in sequences])

    print(f"\n--- {split_name.upper()} SET ---")
    print(f"Articles: {len(lengths):,}")
    print(f"Median token length: {np.median(lengths):.0f}")
    print(f"95th percentile: {np.percentile(lengths, 95):.0f}")
    print(f"99th percentile: {np.percentile(lengths, 99):.0f}")
    print(f"Articles over {MAX_LENGTH} tokens: {(lengths > MAX_LENGTH).sum():,}")
    print(f"Percentage truncated: {(lengths > MAX_LENGTH).mean() * 100:.2f}%")
