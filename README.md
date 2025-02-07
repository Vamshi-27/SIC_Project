# AI-Based Story Generator 📖✨  

This project is an **AI-driven story generation model** trained on the **WritingPrompts dataset**. The model takes a writing prompt as input and generates a creative story using **LSTM-based neural networks**.  

## 🚀 Features  
- ✅ Uses **LSTM** for sequential text generation  
- ✅ Trained on the **WritingPrompts dataset**  
- ✅ Supports **Google Colab (GPU-accelerated training)**  
- ✅ Evaluates performance using **Accuracy & F1 Score**  
- ✅ Deployment using **Streamlit**  

## 🛠 Model Workflow  
1. **Preprocessing** – Tokenizes and pads the dataset  
2. **Training** – LSTM-based neural network (Keras/TensorFlow)  
3. **Evaluation** – Computes **accuracy & F1 score**  
4. **Generation** – Uses trained model to generate stories  
5. **Deployment** – Interactive **Streamlit** web app  

## 📂 File Structure  
├── preprocessing.py # Data processing & tokenization
├── train.py # LSTM model training
├── evaluate.py # Accuracy & F1 score computation
├── generate.py # Story generation script
├── app.py # Streamlit deployment
└── README.md # Project documentation


## 🔧 Setup & Installation  
1. **Clone the repository**  
```bash
git clone https://github.com/your-username/story-generator.git
cd story-generator
Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Train the model (Google Colab recommended)
bash
Copy
Edit
python train.py
Evaluate model performance
bash
Copy
Edit
python evaluate.py
Run the Streamlit app
bash
Copy
Edit
streamlit run app.py
📊 Model Evaluation
The model is evaluated using:

Accuracy – Measures token-level correctness
F1 Score (weighted) – Evaluates overall performance
🌟 Future Improvements
Implement Transformer-based models (GPT, BERT)
Improve dataset preprocessing & filtering
Optimize model training with larger vocab

