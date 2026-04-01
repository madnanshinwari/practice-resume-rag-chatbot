# Resume Chatbot

Ask questions about your resume using **Gemini** + **ChromaDB** + **SentenceTransformers**.

## Project Structure

```
resume_chatbot/
├── config.py        # API keys, model names, chunking + ChromaDB settings
├── embeddings.py    # Gemini embeddings (HuggingFace fallback)
├── ingest.py        # PDF → chunks → embed → store in ChromaDB  (run once)
├── retriever.py     # Query ChromaDB for relevant chunks
├── llm.py           # Gemini LLM + prompt builder
├── chat.py          # chat() pipeline — retrieve → ask Gemini
├── main.py          # CLI entry point
├── resume.pdf       # ← drop your resume here
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/Scripts/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
# Edit .env and paste your GEMINI_API_KEY

# 4. Drop your resume PDF into the project root
in the config.py file replace RESUME_PDF_PATH './resume.pdf'
```

## Usage

```bash
# Step 1 — Index your resume (run once, or re-run when resume changes)
python ingest.py

# Step 2 — Start chatting
python main.py
```

## Embedding Fallback

The project uses **Gemini `text-embedding-004`** by default.
If the Gemini API call fails for any reason, it automatically falls back
to **`all-MiniLM-L6-v2`** via `sentence-transformers` — no code changes needed.
