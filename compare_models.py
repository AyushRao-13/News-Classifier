
import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Configuration
SEED = 42
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load the same training subset used by the CNN
print("Loading AG News dataset...")
dataset = load_dataset("fancyzhx/ag_news")

train_data = dataset["train"].shuffle(seed=SEED).select(range(20000))
test_data = dataset["test"]

X_train = train_data["text"]
y_train = np.array(train_data["label"])

X_test = test_data["text"]
y_test = np.array(test_data["label"])

# 2. Evaluate the existing CNN
print("\nEvaluating existing CNN...")
cnn_model = tf.keras.models.load_model("models/news_cnn_v2.keras")

with open("models/tokenizer_v2.pkl", "rb") as file:
    tokenizer = pickle.load(file)

from tensorflow.keras.preprocessing.sequence import pad_sequences

X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(
    X_test_seq, maxlen=100, padding="post", truncating="post"
)

start = time.perf_counter()
cnn_probabilities = cnn_model.predict(
    X_test_pad, batch_size=64, verbose=0
)
cnn_predictions = np.argmax(cnn_probabilities, axis=1)
cnn_prediction_time = time.perf_counter() - start

# 3. Train the TF-IDF + Logistic Regression baseline
print("\nPreparing TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training Logistic Regression...")
baseline_model = LogisticRegression(
    max_iter=1000,
    random_state=SEED
)

start = time.perf_counter()
baseline_model.fit(X_train_tfidf, y_train)
baseline_training_time = time.perf_counter() - start

start = time.perf_counter()
baseline_predictions = baseline_model.predict(X_test_tfidf)
baseline_prediction_time = time.perf_counter() - start

# 4. Calculate evaluation metrics
def calculate_metrics(name, y_true, y_pred, train_time, prediction_time):
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
        "Training Time (seconds)": train_time,
        "Test Prediction Time (seconds)": prediction_time,
    }

# The CNN's training time was recorded during its original training run,
# so it is not measured again here.
results = pd.DataFrame([
    calculate_metrics(
        "CNN",
        y_test,
        cnn_predictions,
        np.nan,
        cnn_prediction_time
    ),
    calculate_metrics(
        "TF-IDF + Logistic Regression",
        y_test,
        baseline_predictions,
        baseline_training_time,
        baseline_prediction_time
    )
])

print("\n--- Model Comparison ---")
print(results.round(4).to_string(index=False))

results.to_csv(
    os.path.join(RESULTS_DIR, "model_comparison.csv"),
    index=False
)

# 5. Classification reports
for name, predictions in [
    ("CNN", cnn_predictions),
    ("TF-IDF + Logistic Regression", baseline_predictions)
]:
    print(f"\n--- {name}: Classification Report ---")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=LABELS,
        digits=4,
        zero_division=0
    ))

# 6. Plot metric comparison
metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
plot_data = results.set_index("Model")[metrics]

ax = plot_data.T.plot(kind="bar", figsize=(11, 6))
ax.set_title("CNN vs TF-IDF + Logistic Regression")
ax.set_ylabel("Score")
ax.set_xlabel("Evaluation Metric")
ax.set_ylim(0, 1)
ax.legend(title="Model")
plt.xticks(rotation=0)
plt.tight_layout()

chart_path = os.path.join(RESULTS_DIR, "model_comparison.png")
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nComparison table saved to results/model_comparison.csv")
print(f"Comparison chart saved to {chart_path}")
print("Existing CNN model and tokenizer were not modified.")


# 7. Category-wise performance comparison
from sklearn.metrics import precision_recall_fscore_support

print("\n--- Category-wise Model Comparison ---")

category_results = []

for model_name, predictions in [
    ("CNN", cnn_predictions),
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

print(
    category_df.round(4).to_string(index=False)
)

category_df.to_csv(
    os.path.join(RESULTS_DIR, "category_model_comparison.csv"),
    index=False
)

# Plot category-wise F1 scores
f1_pivot = category_df.pivot(
    index="Category",
    columns="Model",
    values="F1-score"
).reindex(LABELS)

ax = f1_pivot.plot(kind="bar", figsize=(10, 6))
ax.set_title("Category-wise F1-score: CNN vs Logistic Regression")
ax.set_ylabel("F1-score")
ax.set_xlabel("News Category")
ax.set_ylim(0, 1)
ax.legend(title="Model")
plt.xticks(rotation=0)
plt.tight_layout()

category_chart = os.path.join(
    RESULTS_DIR, "category_f1_comparison.png"
)
plt.savefig(category_chart, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved category metrics to:")
print("results/category_model_comparison.csv")
print("Saved category F1 chart to:")
print(category_chart)

