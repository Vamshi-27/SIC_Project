import pandas as pd
import tensorflow as tf
import re
import numpy as np
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("🔹 Loading dataset...")
df = pd.read_csv("data/writing_prompts.csv")

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)  # Remove newlines
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)  # Keep common punctuations
    return text.strip()

print("🔹 Cleaning text data...")
df['prompt'] = df['prompt'].astype(str).apply(clean_text)
df['story'] = df['story'].astype(str).apply(clean_text)

# Tokenization
VOCAB_SIZE = 20000  # Adjust this to speed up training if needed

print(f"🔹 Tokenizing text (vocab size: {VOCAB_SIZE})...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['prompt'].tolist() + df['story'].tolist())

# Save tokenizer
tokenizer_path = "models/tokenizer.pkl"
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)

print(f"✅ Tokenizer saved at: {tokenizer_path}")

# Convert text to sequences
print("🔹 Converting text to sequences...")
prompt_sequences = tokenizer.texts_to_sequences(df['prompt'])
story_sequences = tokenizer.texts_to_sequences(df['story'])

# Determine max length efficiently
max_length = max(max(map(len, prompt_sequences)), max(map(len, story_sequences)))

print(f"🔹 Max sequence length: {max_length}")

# Pad sequences
padded_prompts = pad_sequences(prompt_sequences, maxlen=max_length, padding='post')
padded_stories = pad_sequences(story_sequences, maxlen=max_length, padding='post')

# Save processed data
print("🔹 Saving processed data...")
np.save("data/padded_prompts.npy", padded_prompts)
np.save("data/padded_stories.npy", padded_stories)
np.save("data/max_length.npy", max_length)

print("✅ Preprocessing complete! Data saved in 'data/'")
