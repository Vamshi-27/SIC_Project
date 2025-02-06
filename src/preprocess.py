import pandas as pd
import tensorflow as tf
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load dataset
df = pd.read_csv("../data/writing_prompts.csv")

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)  # Remove newlines
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)  # Keep common punctuations
    return text

df['prompt'] = df['prompt'].astype(str).apply(clean_text)
df['story'] = df['story'].astype(str).apply(clean_text)

# Tokenization
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['story'])

# Save tokenizer
with open("../models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df['story'])
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Save processed data
np.save("../data/padded_sequences.npy", padded_sequences)
np.save("../data/max_length.npy", max_length)

print("Preprocessing done!")
