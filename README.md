# AI Story Generator

## Overview
The **AI Story Generator** is a deep learning-based project that generates creative stories based on writing prompts. This project leverages NLP techniques and deep learning models to generate coherent and engaging narratives.

## Features
- **Preprocessing:** Cleans and tokenizes input data from a dataset.
- **Training:** Uses a neural network model to learn from story prompts.
- **Generation:** Produces unique and engaging stories based on input prompts.
- **Evaluation:** Assesses the model's performance on test data.
- **Deployment:** Provides a simple web UI using Streamlit for easy interaction.

## File Structure
```
📁 ai-story-generator
│── 📁 data
│   ├── writing_prompts.csv  # Dataset of writing prompts
│── 📁 models
│   ├── story_generator_model.h5  # Trained model
│── 📁 src
│   ├── preprocess.py  # Data preprocessing script
│   ├── train.py  # Model training script
│   ├── generate.py  # Text generation script
│   ├── evaluate.py  # Model evaluation script
│── 📁 deployment
│   ├── app.py  # Streamlit deployment script
│── requirements.txt  # List of dependencies
│── README.md  # Project documentation
```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Vamshi-27/SIC_Project.git
   cd SIC_Project
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### 1. Preprocess Data
Run the preprocessing script to clean and prepare the dataset:
```sh
python src/preprocess.py
```
### 2. Train the Model
Train the model using the following command:
```sh
python src/train.py
```
### 3. Generate Stories
Generate a new story based on a writing prompt:
```sh
python src/generate.py --prompt "Once upon a time..."
```
### 4. Evaluate Model
Evaluate the trained model on test data:
```sh
python src/evaluate.py
```
### 5. Deploy with Streamlit
Run the Streamlit web app:
```sh
streamlit run deployment/app.py
```

## Dependencies
Ensure you have all required Python packages installed by referring to `requirements.txt`. The project mainly uses:
- TensorFlow/Keras
- NLTK
- Pandas
- NumPy
- Streamlit

## Future Enhancements
- Improve model accuracy with advanced NLP techniques.
- Introduce a feedback mechanism to refine generated stories.
- Deploy as a web application with user authentication.
- Explore integration with voice input for interactive storytelling.
