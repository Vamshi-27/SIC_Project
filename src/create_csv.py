import pandas as pd
import os

# Define dataset path
base_path = "E:/writingPrompts/"  # Update this with your dataset location
datasets = {
    "train": (f"{base_path}train.wp_source", f"{base_path}train.wp_target"),
    "valid": (f"{base_path}valid.wp_source", f"{base_path}valid.wp_target"),
    "test": (f"{base_path}test.wp_source", f"{base_path}test.wp_target"),
}

# Ensure the data directory exists
output_dir = "data/"
os.makedirs(output_dir, exist_ok=True)

# Output CSV file path
csv_path = os.path.join(output_dir, "writing_prompts.csv")

# List to store data
data = []

# Process each dataset
for dataset_type, (source_path, target_path) in datasets.items():
    with open(source_path, "r", encoding="utf-8") as src_file, open(target_path, "r", encoding="utf-8") as tgt_file:
        prompts = src_file.readlines()
        stories = tgt_file.readlines()
        
        # Ensure both files have the same number of lines
        assert len(prompts) == len(stories), f"Mismatch in {dataset_type} files!"
        
        # Store data
        for prompt, story in zip(prompts, stories):
            data.append([prompt.strip(), story.strip(), dataset_type])

# Create DataFrame
df = pd.DataFrame(data, columns=["prompt", "story", "dataset_type"])

# Save as CSV
df.to_csv(csv_path, index=False)

print(f"CSV file created successfully at {csv_path}!")
