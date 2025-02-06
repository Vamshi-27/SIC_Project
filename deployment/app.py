import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

# Load model and tokenizer
model = tf.keras.models.load_model("../models/story_generator_model.h5")

with open("../models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = np.load("../data/max_length.npy")

# Story generation function
def generate_story(seed_text, num_words=100):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_length, padding='post')

        predicted_probs = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted_probs, axis=-1)[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                seed_text += " " + word
                break

    return seed_text

# Streamlit UI
st.title("AI Story Generator")
seed_text = st.text_input("Enter a prompt:", "Once upon a time")
num_words = st.slider("Number of words:", 50, 500, 100)

if st.button("Generate Story"):
    generated_story = generate_story(seed_text, num_words)
    st.write(generated_story)
