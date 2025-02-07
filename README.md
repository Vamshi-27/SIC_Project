# ✨ AI-Based Story Generator  

## 📌 Overview  
The **AI-Based Story Generator** is a web application that generates creative short stories based on user-provided prompts. It uses a **fine-tuned Seq2Seq model** trained on the **Writing Prompts dataset** to produce engaging and contextually relevant narratives.  

This application is designed for **writers, hobbyists, and AI enthusiasts** to experiment with AI-generated storytelling using an intuitive web interface.  

---

## 🎯 Features  
✔️ **AI-Powered Story Generation** – Creates unique, engaging stories from user prompts.  
✔️ **Fine-Tuned Seq2Seq Model** – Improves coherence, creativity, and fluency.  
✔️ **Web-Based Interface** – Built with Streamlit for easy use.  
✔️ **Customizable Creativity Settings** – Adjust temperature, top-k, and top-p.  
✔️ **Story Management** – Save, copy, or share generated stories.  
✔️ **GPU Acceleration Support** – Optional for faster inference.  

---

## 🛠 Setup & Installation  

### 🔹 Clone the Repository  
```bash
git clone https://github.com/vishwanath090/Storygeneration
cd storygenerator
🔹 Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
📂 Dataset: Writing Prompts
The Writing Prompts dataset is a large-scale dataset designed for creative writing and storytelling. It contains diverse prompts paired with user-generated responses, making it an excellent resource for training AI models.

🔹 Dataset Structure
train.wp.source – Training set prompts
train.wp.target – Corresponding training stories
valid.wp.source – Validation set prompts
valid.wp.target – Corresponding validation stories
test.wp.source – Test set prompts
test.wp.target – Corresponding test stories
📌 Model Training & Fine-Tuning
The model is fine-tuned on the Writing Prompts dataset using Hugging Face's Transformers library.

🔹 Training Steps
1️⃣ Preprocess the Dataset – Clean and tokenize the data.
2️⃣ Fine-Tune the Model – Use the command below:

bash
Copy
Edit
python train.py --model_name seq2seq --epochs 3 --batch_size 8
3️⃣ Save the Trained Model – The trained model is stored for later inference.

🚀 Running the Web App
🔹 Start the Application
bash
Copy
Edit
streamlit run app.py
🔹 Access the Web Interface
After running the command, open your browser and navigate to the Streamlit-provided URL.

🔹 Usage
1️⃣ Enter a Writing Prompt – Provide a creative idea or sentence.
2️⃣ Click "Generate" – The AI creates a unique story.
3️⃣ Adjust Creativity Parameters – Modify temperature and top-k settings.
4️⃣ Save, Copy, or Share – Manage generated stories easily.

📊 Model Evaluation
🔹 Performance Metrics
bash
Copy
Edit
📌 Accuracy – Measures token-level correctness  
📌 F1 Score (weighted) – Evaluates overall performance  
🔹 Evaluate Model Performance
bash
Copy
Edit
python evaluate.py
⚙️ Technologies Used
✔️ Backend – Python, TensorFlow/PyTorch, Hugging Face Transformers
✔️ Frontend – Streamlit, HTML, CSS, JavaScript
✔️ Model Architecture – Seq2Seq for text generation
✔️ Deployment – Local execution (future cloud-based options)

🌟 Future Enhancements
bash
Copy
Edit
🚀 Implement Transformer models (GPT, BERT)  
🚀 Improve dataset preprocessing  
🚀 Optimize model training for better results  
🚀 Support multiple story genres for personalized storytelling  
🚀 Develop a mobile-friendly UI  
🚀 Deploy as a cloud-based service  
