
import os
import pickle
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, accuracy_score

# Configuration
VOCAB_SIZE = 15000
MAX_LENGTH = 100
EMBEDDING_DIM = 64
BATCH_SIZE = 64
EPOCHS = 5

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

os.makedirs("models", exist_ok=True)

# 1. Load AG News dataset
print("Loading AG News dataset...")
dataset = load_dataset("fancyzhx/ag_news")

train_data = dataset["train"]
test_data = dataset["test"]

# Use a smaller training subset for faster experimentation
train_data = train_data.shuffle(seed=42).select(range(20000))

X_train = train_data["text"]
y_train = np.array(train_data["label"])

X_test = test_data["text"]
y_test = np.array(test_data["label"])

# 2. NLP preprocessing: tokenize and pad sequences
print("Tokenizing news articles...")

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(
    X_train_seq, maxlen=MAX_LENGTH, padding="post", truncating="post"
)
X_test_pad = pad_sequences(
    X_test_seq, maxlen=MAX_LENGTH, padding="post", truncating="post"
)

# 3. Build the CNN model
print("Building CNN model...")

model = models.Sequential([
    layers.Input(shape=(MAX_LENGTH,)),
    layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=False),
    layers.Conv1D(128, 5, activation="relu"),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 4. Train the model
print("Training CNN model...")

model.fit(
    X_train_pad,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )
    ]
)

# 5. Evaluate on the official test set
print("Evaluating model...")

predictions = np.argmax(model.predict(X_test_pad), axis=1)

print(f"Test Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(classification_report(y_test, predictions, target_names=LABELS))

# 6. Save model and tokenizer
model.save("models/news_cnn.keras")

with open("models/tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

print("Model and tokenizer saved successfully!")
