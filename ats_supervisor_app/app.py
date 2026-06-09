from __future__ import annotations
import io, re, uuid, os
from typing import List, Dict, Tuple
import time
from google.api_core.exceptions import InternalServerError, ServiceUnavailable, DeadlineExceeded

import streamlit as st
from pypdf import PdfReader
from docx import Document

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# -------------------------
# Model: Gemini (free tier)
# -------------------------
import json
import re

def extract_json(text: str):
    """Pull the first JSON object from free-form LLM text."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start:end+1]
    try:
        return json.loads(candidate)
    except Exception:
        # Loose fix: remove trailing commas etc. (keep it simple for now)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            return None
        
def _msg_content(m):
    if isinstance(m, dict):
        return m.get("content") or m.get("text") or str(m)
    # LangChain BaseMessage / Pydantic models
    return getattr(m, "content", None) or getattr(m, "text", None) or str(m)
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("GOOGLE_API_KEY is not set. Create one in Google AI Studio and set the environment variable.")
    st.stop()
model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),  # stable
    temperature=0,
    max_output_tokens=1024,  # avoid huge generations
    # optional: relax safety to avoid odd server errors (can omit)
    safety_settings=None,
)

# -------------------------
# Simple ATS Tool (heuristic)
# -------------------------
_STOP = {
    "and","or","the","a","an","with","in","on","for","to","of","is","are","as",
    "this","that","those","these","be","by","at","from","it","you","your","we","our"
}

def _tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\+\#\.\s]", " ", s)  # keep +/#/. for C++ / C# / .NET
    toks = [t for t in s.split() if t]
    return toks

def _keywords(text: str, limit: int = 100) -> List[str]:
    toks = _tokenize(text)
    seen, out = set(), []
    for t in toks:
        if t in _STOP or len(t) < 2:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= limit:
            break
    return out

def _match(resume_kw: List[str], jd_kw: List[str]) -> Tuple[List[str], List[str]]:
    r = set(resume_kw)
    matched = [k for k in jd_kw if k in r]
    missing = [k for k in jd_kw if k not in r]
    return matched, missing

def ats_score(resume: str, job_description: str = "") -> Dict:
    """
    Compute an ATS-style keyword-overlap score between a resume and an optional job description.

    Args:
        resume: Plaintext resume content (already extracted from PDF/DOCX/TXT).
        job_description: Optional JD text. If omitted/empty, we score self-coverage.

    Returns:
        dict with:
          - score (float, 0–100)
          - matched_keywords (list[str])
          - missing_keywords (list[str])
          - resume_keywords_sample (list[str])
          - jd_keywords_sample (list[str])
          - notes (str)
    """
    resume_kw = _keywords(resume, limit=120)
    jd_kw = _keywords(job_description, limit=80) if job_description else resume_kw
    matched, missing = _match(resume_kw, jd_kw)
    total = max(len(jd_kw), 1)
    score = round(100.0 * len(matched) / total, 1)
    return {
        "score": score,
        "matched_keywords": matched[:60],
        "missing_keywords": missing[:60],
        "resume_keywords_sample": resume_kw[:40],
        "jd_keywords_sample": jd_kw[:40],
        "notes": "Heuristic overlap only. Improve with weights, synonyms, and section-aware parsing."
    }

# -------------------------
# Agents
# -------------------------
smalltalk_agent = create_react_agent(
    model=model,
    tools=[],
    name="smalltalk_agent",
    prompt=(
        "You are a friendly small-talk assistant. Keep replies short and warm. "
        "Do not call tools."
    ),
)

ats_agent = create_react_agent(
    model=model,
    tools=[ats_score],
    name="ats_agent",
    prompt=(
        "You evaluate resumes. The user will provide a resume (text extracted from a file) "
        "and optionally a job description.\n"
        "- You MUST call ats_score(resume, job_description) exactly once.\n"
        "- Return ONLY the JSON returned by the tool. Do not add any extra text."
    ),
)
# -------------------------
# Supervisor (routing)
# -------------------------
supervisor_prompt = (
    "You are the supervisor for two agents:\n"
    "- smalltalk_agent: general chat.\n"
    "- ats_agent: resume/ATS analysis.\n"
    "Routing rules:\n"
    "1) If the user provides a resume or asks for ATS scoring/fit/JD comparison → ats_agent.\n"
    "2) Otherwise → smalltalk_agent.\n"
    "Allow multiple handoffs until the task is fully solved. Keep answers concise."
)

workflow = create_supervisor(
    [smalltalk_agent, ats_agent],
    model=model,
    prompt=supervisor_prompt,
    output_mode="last_message",
)

# Memory (short-term convos + long-term store)
checkpointer = InMemorySaver()
store = InMemoryStore()
app = workflow.compile(checkpointer=checkpointer, store=store)
def invoke_safe(payload, thread_id: str, retries: int = 2):
    """
    Call app.invoke with a small retry/backoff to ride out transient 500s from Gemini.
    Retries on InternalServerError / ServiceUnavailable / DeadlineExceeded.
    """
    for i in range(retries + 1):
        try:
            return app.invoke(payload, config={"configurable": {"thread_id": thread_id}})
        except (InternalServerError, ServiceUnavailable, DeadlineExceeded) as e:
            if i < retries:
                time.sleep(1.5 * (i + 1))  # backoff: 1.5s, 3.0s, ...
                continue
            raise
        except Exception as e:
            # Fallback: retry once/twice if it looks like a 500-ish server error
            msg = str(e)
            if (("InternalServerError" in msg) or (" 500 " in msg) or ("code=500" in msg)) and i < retries:
                time.sleep(1.5 * (i + 1))
                continue
            raise

# -------------------------
# File parsing
# -------------------------
def read_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(text_parts)
    except Exception as e:
        return f"__ERROR__: Failed to read PDF: {e}"

def read_docx(file_bytes: bytes) -> str:
    try:
        bio = io.BytesIO(file_bytes)
        doc = Document(bio)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return f"__ERROR__: Failed to read DOCX: {e}"

def read_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"__ERROR__: Failed to read TXT: {e}"

def read_any(file) -> str:
    name = (file.name or "").lower()
    data = file.read()
    if name.endswith(".pdf"):
        return read_pdf(data)
    if name.endswith(".docx"):
        return read_docx(data)
    if name.endswith(".txt"):
        return read_txt(data)
    return "__ERROR__: Unsupported file type. Please upload PDF, DOCX, or TXT."

def clean_text(s: str, max_chars: int = 30000) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s[:max_chars]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="ATS Scorer ", page_icon="🤖")
st.title("🤖 ATS Scorer")
import uuid


st.caption("Upload a resume (PDF/DOCX/TXT), optional JD, and get an ATS-style score. Memory is enabled per session.")

# Stable thread_id for memory per browser session
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"thread-{uuid.uuid4()}"

# Reset button (clears memory by creating a new thread_id)
if st.button("Reset session"):
    st.session_state["thread_id"] = f"thread-{uuid.uuid4()}"
    st.success("Session reset. Try again.")

# (Optional) show current session id
st.caption(f"Session thread: {st.session_state['thread_id']}")
tabs = st.tabs(["ATS Score", "Chat"])

with tabs[0]:
    st.subheader("Upload Resume (+ optional JD)")
    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader("Resume (PDF/DOCX/TXT)", type=["pdf","docx","txt"], key="resume")
    with col2:
        jd_mode = st.radio("Job Description input", ["Text", "Upload file"], horizontal=True)
    jd_text = ""
    if jd_mode == "Text":
        jd_text = st.text_area("Job Description (optional)", height=140, placeholder="Paste JD here...")
    else:
        jd_file = st.file_uploader("JD file (PDF/DOCX/TXT)", type=["pdf","docx","txt"], key="jd")
        if jd_file:
            jd_text = read_any(jd_file)

    if st.button("Compute ATS Score", type="primary", use_container_width=True):
        if not resume_file:
            st.error("Please upload a resume file.")
        else:
            resume_text = read_any(resume_file)
            if resume_text.startswith("__ERROR__"):
                st.error(resume_text.replace("__ERROR__:", "").strip())
            else:
                # Clean + clamp input sizes
                resume_text = clean_text(resume_text)
                jd_text_clean = clean_text(jd_text) if jd_text else ""

                # 1) Always compute a deterministic score locally
                data = ats_score(resume=resume_text, job_description=jd_text_clean)

                # 2) Optionally ask the supervisor/agents ONLY for short tips (no JSON),
                #    so we don't depend on the LLM to produce the score.
                tips = ""
                try:
                    tips_prompt = (
                        "You are an ATS advisor. Based on the following resume"
                        + (" and job description" if jd_text_clean else "")
                        + ", give exactly 3 short improvement tips (bulleted). "
                        "Do NOT include JSON and do NOT restate the resume/JD.\n\n"
                        "Resume:\n" + resume_text + ("\n\nJob Description:\n" + jd_text_clean if jd_text_clean else "")
                    )
                    result = invoke_safe(
                        {"messages": [{"role": "user", "content": tips_prompt}]},
                        thread_id=st.session_state.thread_id
                    )
                    last = result["messages"][-1]
                    tips = _msg_content(last) or ""
                except Exception as _:
                    # Tips are optional—ignore failures.
                    tips = ""

                # 3) Display score + JSON + optional tips
                st.markdown("### Result")
                st.metric("ATS Score", f"{data.get('score', 0)} / 100")

                st.markdown("#### Report (JSON)")
                st.json(data)
                st.download_button(
                    "Download JSON report",
                    data=json.dumps(data, indent=2).encode("utf-8"),
                    file_name="ats_report.json",
                    mime="application/json",
                    use_container_width=True
                )

                if tips.strip():
                    st.markdown("#### Improvement Tips")
                    st.write(tips)


with tabs[1]:
    st.subheader("Small Talk (memory on this tab too)")
    user_chat = st.text_input("Say something…", placeholder="Hi! Can you review my resume later?")
    if st.button("Send", use_container_width=True):
        if user_chat.strip():
            result = invoke_safe(
                {"messages": [{"role": "user", "content": user_chat}]},
                thread_id=st.session_state.thread_id
            )

            last = result["messages"][-1]
            st.markdown("**Assistant:**")
            st.write(_msg_content(last))

st.divider()
with st.expander("What’s hardcoded?"):
    st.markdown("""
- **Model**: `gemini-2.5-flash` with `temperature=0`
- **File types**: PDF/DOCX/TXT (no `.doc`/images/OCR)
- **ATS method**: simple keyword-overlap heuristic
- **Routing**: supervisor rules that send uploads/ATS asks to `ats_agent`, everything else to `smalltalk_agent`
- **Memory**: in-process `InMemorySaver` + `InMemoryStore` keyed by a session `thread_id`
""")
