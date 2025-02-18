from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Load the input prompts and actual stories
print("Loading input prompts...")
padded_prompts = np.load("data/padded_prompts.npy")  # Input prompts
print("Loading actual generated stories...")
padded_stories = np.load("data/padded_stories.npy")  # Actual generated stories

# y_true: The actual next words (shifted left)
print("Processing y_true by removing the first token...")
y_true = padded_stories[:, 1:]  # Remove the first token to align with predictions

# y_pred: Mock predictions (shifted left for comparison)
print("Generating mock predictions by shifting left...")
y_pred = np.roll(padded_stories, -1, axis=1)[:, :-1]  # Shift left and remove last token

# Convert to 1D for metric calculation
print("Flattening y_true and y_pred for metric calculations...")
y_true = y_true.flatten()
y_pred = y_pred.flatten()

# Compute accuracy and F1 score
print("Computing accuracy and F1 score...")
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
