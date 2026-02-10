# 🏥 MedAI – AI Powered Medical Chatbot (FastAPI + RAG)

MedAI is an **AI-powered medical assistant chatbot** inspired by ChatGPT, built using **FastAPI**, **LangChain**, and **Retrieval-Augmented Generation (RAG)**.  
It provides intelligent, document-based medical responses by combining **LLMs**, **vector search**, and **medical knowledge sources**.

> ⚠️ **Disclaimer**: This project is for educational and research purposes only. It does **not** replace professional medical advice.

---

## 🚀 Features

- 🤖 AI-powered medical chatbot (ChatGPT-style)
- 📚 Retrieval-Augmented Generation (RAG)
- 🧠 Context-aware responses using medical documents
- 📄 Supports PDF & text-based medical knowledge
- 💬 Chat history management
- 🔐 Secure environment configuration
- 🔊 Optional voice interaction module
- ⚡ Fast & scalable backend with FastAPI

---

## 🧠 Tech Stack

### Backend
- **Python**
- **FastAPI**
- **LangChain**
- **MongoDB** (chat history & sessions)
- **Firebase** (authentication / integration)

### AI & RAG
- Embedding models (via LangChain)
- Vector Store for semantic search
- Custom retriever pipeline

### Frontend
- Basic HTML interface
- Node.js dependencies (optional)

## 📁 Project Structure
medical_chatbot/
│
├── main.py # FastAPI entry point
├── database.py # MongoDB connection
├── feedback_system.py # User feedback handling
├── langchain_cache.py # LLM caching
│
├── rag/ # RAG core modules
│ ├── document_loader.py
│ ├── embeddings.py
│ ├── retriever.py
│ ├── vector_store.py
│ └── voice_module.py
│
├── routers/ # API routes
│ ├── auth_router.py
│ ├── chatbot_router.py
│ ├── firebase_router.py
│ └── rag_router.py
│
├── schemas/ # Pydantic models
│ └── model.py
│
├── data/
│ └── medical_docs/ # Medical documents (PDF/TXT)
│
├── requirements.txt # Python dependencies
├── package.json # Frontend dependencies
├── .gitignore
└── README.md

---

## ⚙️ Installation & Setup
### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Medical-Chatbot.git
cd medical_chatbot

2️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Environment variables
Create a .env file (not committed to GitHub):
MONGODB_URI=your_mongodb_uri
OPENAI_API_KEY=your_api_key
FIREBASE_PROJECT_ID=your_project_id

##▶️ Run the Application
uvicorn main:app --reload
Then open:
http://127.0.0.1:8000
📌 API Highlights
/chat – Chat with the medical assistant
/rag/query – RAG-based document search
/history – Retrieve chat history
/auth – Authentication route
🔐 Security Notes
.env files are ignored
Firebase service account keys are never committed
MongoDB local data is excluded from GitHub
📈 Future Improvements
🧑‍⚕️ Disease-specific fine-tuned models
📊 Admin dashboard
🌐 Full frontend (React / Next.js)
🧾 Medical citation sources
🗣️ Advanced voice assistant
👨‍💻 Author
Tayyab Aslam
AI Engineer
📍 Azad Kashmir, Pakistan
⭐ Support
If you like this project:
⭐ Star the repository
🐛 Report issues
🤝 Contribute improvements
Built with ❤️ using FastAPI, LangChain, and AI

