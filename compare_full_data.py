
import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

SEED = 42
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
os.makedirs("results", exist_ok=True)

# 1. Load the full dataset
print("Loading AG News...")
dataset = load_dataset("fancyzhx/ag_news")
train_data = dataset["train"].shuffle(seed=SEED)
test_data = dataset["test"]

X_train = train_data["text"]
y_train = np.array(train_data["label"])
X_test = test_data["text"]
y_test = np.array(test_data["label"])

print(f"Training articles: {len(X_train):,}")
print(f"Test articles: {len(X_test):,}")

# 2. Evaluate saved v3 CNN
print("\nLoading full-data CNN v3...")
cnn = tf.keras.models.load_model("models/news_cnn_v3.keras")

with open("models/tokenizer_v3.pkl", "rb") as f:
    tokenizer = pickle.load(f)

X_test_pad = pad_sequences(
    tokenizer.texts_to_sequences(X_test),
    maxlen=100,
    padding="post",
    truncating="post",
)

start = time.perf_counter()
cnn_probabilities = cnn.predict(
    X_test_pad, batch_size=64, verbose=1
)
cnn_predictions = np.argmax(cnn_probabilities, axis=1)
cnn_inference_time = time.perf_counter() - start

# 3. Train TF-IDF + Logistic Regression on all 120K articles
print("\nCreating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training Logistic Regression on full training set...")
baseline = LogisticRegression(
    max_iter=1000,
    random_state=SEED,
)

start = time.perf_counter()
baseline.fit(X_train_tfidf, y_train)
baseline_training_time = time.perf_counter() - start

start = time.perf_counter()
baseline_predictions = baseline.predict(X_test_tfidf)
baseline_inference_time = time.perf_counter() - start

# 4. Compare metrics
def get_metrics(name, y_true, y_pred, inference_time, training_time=None):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Training Time (seconds)": training_time,
        "Test Prediction Time (seconds)": inference_time,
    }

results = pd.DataFrame([
    get_metrics(
        "CNN v3 (120K)",
        y_test,
        cnn_predictions,
        cnn_inference_time,
    ),
    get_metrics(
        "TF-IDF + Logistic Regression (120K)",
        y_test,
        baseline_predictions,
        baseline_inference_time,
        baseline_training_time,
    ),
])

print("\n--- Full-Data Model Comparison ---")
print(results.round(4).to_string(index=False))

results.to_csv(
    "results/full_data_model_comparison.csv",
    index=False,
)

# 5. Category-wise reports
for name, predictions in [
    ("CNN v3", cnn_predictions),
    ("TF-IDF + Logistic Regression", baseline_predictions),
]:
    print(f"\n--- {name} Classification Report ---")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=LABELS,
        digits=4,
        zero_division=0,
    ))

# 6. Save a metric comparison chart
metric_columns = [
    "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"
]

ax = results.set_index("Model")[metric_columns].T.plot(
    kind="bar", figsize=(11, 6)
)

# Dark theme for NewsSense AI
fig = ax.get_figure()
fig.patch.set_facecolor("#0b1020")
ax.set_facecolor("#151f31")

ax.set_title(
    "CNN v3 vs Logistic Regression — Full Training Data",
    color="white",
    pad=15
)
ax.set_ylabel("Score", color="white")
ax.set_xlabel("Metric", color="white")
ax.set_ylim(0, 1)

ax.tick_params(colors="white")
ax.legend(
    title="Model",
    facecolor="#151f31",
    edgecolor="#334155",
    labelcolor="white"
)

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "results/full_data_model_comparison.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)
plt.show()


print("\nSaved results/full_data_model_comparison.csv")
print("Saved results/full_data_model_comparison.png")
print("Existing saved CNN models were not modified.")


from sklearn.metrics import precision_recall_fscore_support

category_results = []

for model_name, predictions in [
    ("CNN v3", cnn_predictions),
    ("TF-IDF + Logistic Regression", baseline_predictions)
]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        zero_division=0
    )

    for i, category in enumerate(LABELS):
        category_results.append({
            "Category": category,
            "Model": model_name,
            "Precision": precision[i],
            "Recall": recall[i],
            "F1-score": f1[i],
            "Support": support[i]
        })

category_df = pd.DataFrame(category_results)

print("\n--- Full-Data Category-wise Comparison ---")
print(category_df.round(4).to_string(index=False))

category_df.to_csv(
    "results/full_data_category_comparison.csv",
    index=False
)

# Plot category-wise F1 scores
f1_pivot = category_df.pivot(
    index="Category",
    columns="Model",
    values="F1-score"
).reindex(LABELS)

ax = f1_pivot.plot(kind="bar", figsize=(10, 6))
ax.set_title(
    "Category-wise F1-score: CNN v3 vs Logistic Regression",
    color="white",
    pad=15
)
ax.set_ylabel("F1-score", color="white")
ax.set_xlabel("News Category", color="white")
ax.set_ylim(0, 1)

ax.tick_params(colors="white")

legend = ax.legend(title="Model", frameon=True)
legend.get_frame().set_facecolor("#151f31")
legend.get_frame().set_edgecolor("#334155")
legend.get_title().set_color("white")

for text in legend.get_texts():
    text.set_color("white")

fig = ax.get_figure()
fig.patch.set_facecolor("#0b1020")
ax.set_facecolor("#151f31")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "results/full_data_category_f1.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)
plt.show()


print("\nSaved: results/full_data_category_comparison.csv")
print("Saved: results/full_data_category_f1.png")

