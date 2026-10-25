
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Create output folder
os.makedirs("results/eda", exist_ok=True)

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

print("Loading AG News dataset...")
dataset = load_dataset("fancyzhx/ag_news")

# Use the complete training split for EDA
df = pd.DataFrame(dataset["train"])
df["category"] = df["label"].map(dict(enumerate(LABELS)))

print("\n--- Dataset Overview ---")
print(f"Number of articles: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")
print("\nSample articles:")
print(df[["category", "text"]].head())

# 1. Missing values and duplicates
print("\n--- Data Quality ---")
print("Missing values:")
print(df[["text", "label"]].isnull().sum())
print(f"Duplicate article texts: {df['text'].duplicated().sum():,}")

# 2. Category distribution
category_counts = df["category"].value_counts().reindex(LABELS)

print("\n--- Category Distribution ---")
print(category_counts)

plt.figure(figsize=(9, 5))
category_counts.plot(kind="bar")
plt.title("AG News Training Set: Category Distribution")
plt.xlabel("News Category")
plt.ylabel("Number of Articles")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("results/eda/category_distribution.png", dpi=300)
plt.show()

# 3. Text length analysis
df["word_count"] = df["text"].apply(lambda text: len(text.split()))
df["character_count"] = df["text"].str.len()

print("\n--- Text Length Statistics ---")
print(df.groupby("category")[["word_count", "character_count"]].describe().round(2))

plt.figure(figsize=(9, 5))
df.boxplot(column="word_count", by="category", grid=False)
plt.title("Article Word Counts by Category")
plt.suptitle("")
plt.xlabel("News Category")
plt.ylabel("Number of Words")
plt.tight_layout()
plt.savefig("results/eda/text_length_by_category.png", dpi=300)
plt.show()

plt.figure(figsize=(9, 5))
df["word_count"].hist(bins=40)
plt.title("Distribution of Article Word Counts")
plt.xlabel("Number of Words")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.savefig("results/eda/word_count_distribution.png", dpi=300)
plt.show()

# 4. Vocabulary and frequent words
def tokenize_words(text):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())

all_words = [
    word
    for text in df["text"]
    for word in tokenize_words(text)
]

print(f"\nTotal word tokens: {len(all_words):,}")
print(f"Unique word types: {len(set(all_words)):,}")

# Basic common-word analysis; includes stop words
common_words = Counter(all_words).most_common(20)
words_df = pd.DataFrame(common_words, columns=["word", "count"])

print("\n--- Top 20 Words Overall ---")
print(words_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(words_df["word"][::-1], words_df["count"][::-1])
plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig("results/eda/top_words.png", dpi=300)
plt.show()

# 5. Most frequent words within each category
for category in LABELS:
    category_texts = df.loc[df["category"] == category, "text"]
    words = [
        word
        for text in category_texts
        for word in tokenize_words(text)
    ]
    print(f"\nTop 10 words in {category}:")
    print(Counter(words).most_common(10))

print("\nEDA complete. Charts saved in results/eda/")


# 6. Category-specific informative words (stop words removed)

print("\n--- Category-Specific Informative Words ---")

def informative_words(text):
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    return [
        word for word in words
        if word not in ENGLISH_STOP_WORDS
    ]

category_word_counts = {}

for category in LABELS:
    texts = df.loc[df["category"] == category, "text"]
    words = []

    for text in texts:
        words.extend(informative_words(text))

    category_word_counts[category] = Counter(words)

    print(f"\nTop 15 informative words in {category}:")
    print(category_word_counts[category].most_common(15))

# Visualize the top 10 words in each category
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, category in zip(axes, LABELS):
    top_words = category_word_counts[category].most_common(10)

    words = [word for word, count in top_words][::-1]
    counts = [count for word, count in top_words][::-1]

    ax.barh(words, counts)
    ax.set_title(category)
    ax.set_xlabel("Word frequency")

fig.suptitle(
    "Top Words by News Category (Stop Words Removed)"
)
fig.tight_layout()

output_path = "results/eda/category_specific_words.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nCategory-specific word chart saved to: {output_path}")
