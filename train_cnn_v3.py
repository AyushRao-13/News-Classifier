
import os
import pickle
import time
import numpy as np
import tensorflow as tf

from datasets import load_dataset
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report

# Configuration
SEED = 42
VOCAB_SIZE = 15000
MAX_LENGTH = 100
EMBEDDING_DIM = 64
BATCH_SIZE = 64
EPOCHS = 8

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

tf.keras.utils.set_random_seed(SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. Load the complete AG News training set
print("Loading the complete AG News dataset...")
dataset = load_dataset("fancyzhx/ag_news")

train_data = dataset["train"].shuffle(seed=SEED)
test_data = dataset["test"]

X_train = train_data["text"]
y_train = np.array(train_data["label"])

X_test = test_data["text"]
y_test = np.array(test_data["label"])

print(f"Training articles: {len(X_train):,}")
print(f"Test articles: {len(X_test):,}")

# 2. Fit tokenizer using training data only
print("Tokenizing text...")
tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)
tokenizer.fit_on_texts(X_train)

X_train_pad = pad_sequences(
    tokenizer.texts_to_sequences(X_train),
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    tokenizer.texts_to_sequences(X_test),
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

# 3. Build parallel-filter CNN
inputs = layers.Input(shape=(MAX_LENGTH,))

embedding = layers.Embedding(
    VOCAB_SIZE,
    EMBEDDING_DIM,
    mask_zero=False
)(inputs)

branch_3 = layers.Conv1D(
    96, 3, padding="same", activation="relu"
)(embedding)
branch_3 = layers.GlobalMaxPooling1D()(branch_3)

branch_5 = layers.Conv1D(
    96, 5, padding="same", activation="relu"
)(embedding)
branch_5 = layers.GlobalMaxPooling1D()(branch_5)

branch_7 = layers.Conv1D(
    96, 7, padding="same", activation="relu"
)(embedding)
branch_7 = layers.GlobalMaxPooling1D()(branch_7)

combined = layers.Concatenate()(
    [branch_3, branch_5, branch_7]
)

x = layers.Dense(128, activation="relu")(combined)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(4, activation="softmax")(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 4. Train with validation and adaptive learning rate
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=0.00001,
        verbose=1
    )
]

print("\nTraining on the full dataset...")
start_time = time.perf_counter()

history = model.fit(
    X_train_pad,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

training_time = time.perf_counter() - start_time

# 5. Evaluate once on the official test set
print("\nEvaluating on the official test set...")
probabilities = model.predict(
    X_test_pad,
    batch_size=64,
    verbose=1
)

predictions = np.argmax(probabilities, axis=1)
accuracy = accuracy_score(y_test, predictions)

print(f"\nTraining time: {training_time:.2f} seconds")
print(f"Full-data CNN accuracy: {accuracy:.4f}")

print(classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2, 3],
    target_names=LABELS,
    digits=4
))

# 6. Save separately; do not overwrite previous models
model.save("models/news_cnn_v3.keras")

with open("models/tokenizer_v3.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

print("\nSaved model: models/news_cnn_v3.keras")
print("Saved tokenizer: models/tokenizer_v3.pkl")
print("Previous models remain unchanged.")
