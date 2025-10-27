# Agentic Resume Bot - Minimal Runnable Prototype

# 🧠 Agentic Resume Bot  

An AI-powered resume analysis and matching assistant that helps users optimize their resumes for specific job descriptions using **LLMs**, **semantic search**, and **vector embeddings**.

---

## 🚀 Overview  

The **Agentic Resume Bot** intelligently analyzes resumes and job descriptions to:  
- Identify matching and missing skills  
- Suggest improvements to increase job match  
- Compute similarity scores using embeddings  
- Provide an AI-generated summary of fit  

---

## 🧩 Features  

✅ Resume text extraction  
✅ Job description parsing  
✅ Similarity score and skill gap analysis  
✅ AI-based suggestions  
✅ Simple Streamlit interface  

---

## 🏗️ Tech Stack  

| Category | Tools / Libraries |
|-----------|-------------------|
| Programming | Python |
| AI/ML | LangChain, OpenAI API, SentenceTransformers, FAISS |
| Frontend | Streamlit |
| Data Handling | Pandas, NumPy |
| Environment | Virtualenv |
| Version Control | Git, GitHub |

---

## ⚙️ Setup Instructions  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/AngelaSuperna29/agentic_resume_bot.git
cd agentic_resume_bot

###2️⃣ Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your API key

Create a .env file in the root folder and add:

OPENAI_API_KEY=your_api_key_here


5️⃣ Run the app
streamlit run app.py
