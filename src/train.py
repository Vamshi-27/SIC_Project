import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import pickle

# Load processed data
padded_sequences = np.load("../data/padded_sequences.npy")
max_length = np.load("../data/max_length.npy")

# Load tokenizer
with open("../models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1

# Model architecture
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    LSTM(256, return_sequences=True),
    LSTM(256),
    Dense(128, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
y = np.roll(padded_sequences, -1, axis=1)  # Shifted labels
model.fit(padded_sequences, y, epochs=10, batch_size=64)

# Save model
model.save("../models/story_generator_model.h5")

print("Training completed!")
