# PrepMind – DSA AI Chatbot

PrepMind is an AI-powered Data Structures & Algorithms assistant built using Retrieval-Augmented Generation (RAG).  
It helps students understand DSA concepts, generate optimized code, and analyze time and space complexity.

---

## 🚀 Features

- 📘 Concept explanations (clear and structured)
- 💻 Code generation for DSA problems
- 📊 Time & Space complexity analysis
- 🔍 Retrieval-based answers using FAISS
- ⚡ Powered by Groq LLM
- 🖥️ Interactive Streamlit UI

---

## 🏗️ Tech Stack

- Python
- Streamlit
- FAISS (Vector Search)
- Groq LLM API
- python-dotenv
- Retrieval-Augmented Generation (RAG)

---

## 📂 Project Structure
PrepMind/
│
├── app.py # Streamlit UI
├── ingestion.py # Data processing & embeddings
├── llm_pipeline.py # LLM query + retrieval logic
├── vector_store.py # FAISS index handling
├── requirements.txt
├── .env.example
├── faiss_indices/
└── logs/

---

## ⚙️ Setup Instructions (Run Locally)

### 1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/PrepMind-DSA-Chatbot.git

cd PrepMind-DSA-Chatbot

### 2️⃣ Create virtual environment

Windows:
python -m venv venv
venv\Scripts\activate


Mac/Linux:

python3 -m venv venv
source venv/bin/activate


### 3️⃣ Install dependencies


pip install -r requirements.txt


### 4️⃣ Add Environment Variables

Create a `.env` file and add:


GROQ_API_KEY=your_api_key_here


### 5️⃣ Run the application


streamlit run app.py


App will open at:
http://localhost:8501

---

## 🧠 Example Questions

- Explain why quicksort has O(n log n) average complexity.
- Design an LRU cache with O(1) operations.
- Why does Dijkstra fail with negative weights?

---

## 🔐 Security Note

Do NOT upload your `.env` file or API keys to GitHub.  
Use environment variables for all secrets.

---

## 📌 Future Improvements

- Add user authentication
- Deploy on Streamlit Cloud
- Improve retrieval quality
- Add more DSA topic coverage

---

## 👤 Author

Your Name  
B.Tech CSE  

IMPORTANT
Replace:

YOUR_USERNAME
Your Name

with your actual details.
