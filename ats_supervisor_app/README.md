# ATS Scorer — Gemini + LangGraph Supervisor (with Memory)

This is a minimal Streamlit app that uses **Gemini (free tier)** + **LangGraph Supervisor** to route between:
- `ats_agent` for resume scoring
- `smalltalk_agent` for general chat

It supports **PDF/DOCX/TXT** uploads for resumes and optional JD, and uses in-memory checkpoints to keep conversation context per browser session.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

# Gemini free-tier key:
# Windows PowerShell: $env:GOOGLE_API_KEY="YOUR_KEY"
export GOOGLE_API_KEY="YOUR_KEY"
```

## Run
```bash
streamlit run app.py
```

## Notes
- The ATS scoring is a simple **keyword overlap heuristic**. Improve by adding weights, synonyms, and section-aware parsing.
- Memory uses `InMemorySaver` and `InMemoryStore`. For persistence across restarts, swap to a database/SQLite saver.
