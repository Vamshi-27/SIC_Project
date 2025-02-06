from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Load predictions and actual values
y_true = np.load("../data/padded_sequences.npy")[:, 1:]  # Shifted actual values
y_pred = np.roll(y_true, -1, axis=1)  # Mock predictions

# Convert to 1D
y_true = y_true.flatten()
y_pred = y_pred.flatten()

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
