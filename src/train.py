import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import pickle
import os

# Set up paths for VS Code execution
data_dir = os.path.join(os.getcwd(), "data")
models_dir = os.path.join(os.getcwd(), "models")

# Load processed data
print("Loading processed data...")
padded_prompts = np.load(os.path.join(data_dir, "padded_prompts.npy"))
padded_stories = np.load(os.path.join(data_dir, "padded_stories.npy"))
max_length = np.load(os.path.join(data_dir, "max_length.npy"))

# Load tokenizer
print("Loading tokenizer...")
with open(os.path.join(models_dir, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1

# ✅ Fix target labels: Shift sequences for next-word prediction
X = padded_prompts  # Inputs: Prompt sequences
y = np.roll(padded_stories, shift=-1, axis=1)  # Targets: Shifted stories
y = y[:, :-1]  # Remove last column to match input shape

print(f"Data loaded. Vocab size: {vocab_size}, Max sequence length: {max_length}")

# ✅ Fix model architecture
print("Building the model...")
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    LSTM(256, return_sequences=True),
    LSTM(256, return_sequences=True),  # Ensure LSTM outputs sequence
    Dense(vocab_size, activation='softmax')  # Output per timestep
])

# ✅ Use categorical_crossentropy instead of sparse_categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ✅ Convert y to one-hot encoding (needed for categorical_crossentropy)
print("Converting target labels to one-hot encoding...")
y_one_hot = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

print("Starting training...")
# Training
for epoch in range(10):
    print(f"Starting epoch {epoch + 1}...")
    history = model.fit(X, y_one_hot, epochs=1, batch_size=32, verbose=1)
    loss = history.history['loss'][0]
    accuracy = history.history['accuracy'][0]
    print(f"Epoch {epoch + 1} completed. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")

# Save model
print("Saving the trained model...")
os.makedirs(models_dir, exist_ok=True)
model.save(os.path.join(models_dir, "story_generator_model.h5"))       # and save the model as .keras format also

print("✅ Training completed and model saved!")
