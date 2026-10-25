import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)

# Configuration
MAX_LENGTH = 100
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# Load the saved model
print("Loading trained CNN model...")
model = tf.keras.models.load_model("models/news_cnn_v3.keras")

# Load the saved tokenizer
with open("models/tokenizer_v3.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Load the official test dataset
print("Loading AG News test dataset...")
test_data = load_dataset("fancyzhx/ag_news", split="test")

X_test = test_data["text"]
y_test = np.array(test_data["label"])

# Apply the same preprocessing used during training
print("Preprocessing test articles...")
sequences = tokenizer.texts_to_sequences(X_test)

X_test_pad = pad_sequences(
    sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

# Predict categories
print("Generating predictions...")
probabilities = model.predict(X_test_pad, batch_size=64, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

# Print evaluation metrics
print("\nTest Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=LABELS,
        digits=4
    )
)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

os.makedirs("results", exist_ok=True)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABELS
)

# Create the figure
fig, ax = plt.subplots(figsize=(9, 7))

# Match the NewsSense AI dark theme
fig.patch.set_facecolor("#0b1020")
ax.set_facecolor("#151f31")

# Draw the confusion matrix
display.plot(
    ax=ax,
    values_format="d",
    xticks_rotation=25,
    colorbar=False,
    cmap="viridis"
)

# Style labels and title
ax.set_title(
    "AG News Classification - Full-Data CNN v3",
    color="white",
    pad=18
)
ax.set_xlabel("Predicted label", color="white")
ax.set_ylabel("True label", color="white")
ax.tick_params(colors="white")

# Make the cell values readable
for text in ax.texts:
    text.set_color("white")
    text.set_fontsize(11)

fig.tight_layout()

# Save the figure
fig.savefig(
    "results/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

plt.show()

print("\nConfusion matrix saved to results/confusion_matrix.png")
