# Energy Operations Document RAG Assistant (Greenko Group Assignment)

A grounded Retrieval-Augmented Generation (RAG) assistant built for renewable-energy operations, safety, and maintenance Standard Operating Procedures (SOPs).

## Architecture & Flow
1. **Document Ingestion & Chunking (`backend/rag_engine.py`)**: Loads bundled SOP markdown documents (`data/*.md`) and creates both section-level and sentence-level chunks tagged with source filename metadata.
2. **Hybrid Retrieval**: Uses TF-IDF (1–2 n-gram) cosine similarity combined with technical keyword overlap scoring to retrieve top-k relevant SOP chunks with exact similarity scores.
3. **Grounded Answer Generation**:
   - Uses **Gemini 2.5 Flash** (`google-genai`) with strict citation-enforced prompting when `GEMINI_API_KEY` is set.
   - Automatically falls back to deterministic **Extractive Grounded Synthesis** when running offline or without an API key, ensuring zero downtime.
4. **Hallucination Guardrails**:
   - Enforces a similarity threshold (`0.29`) to abstain on out-of-scope queries (`ABSTAINED_OUT_OF_SCOPE`).
   - Verifies inline `[source.md]` citations and measures lexical token overlap against retrieved evidence.
5. **FastAPI + Frontend (`backend/app.py`, `frontend/index.html`)**: Serves both the REST API (`/api/demo`, `/api/documents`, `/api/run`) and an interactive web UI on a single port.

## Local Setup & Run Commands
```bash
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python evaluate.py
uvicorn backend.app:app --reload --port 8000
```
Open `http://127.0.0.1:8000` in your browser.

## Evaluation Results (`evaluate.py`)
- **Retrieval Recall@1 (In-scope SOPs):** 100% (6/6)
- **Citation Verification Rate:** 100% (6/6)
- **Out-of-Scope Abstention Rate (Hallucination Prevention):** 100% (2/2)

## Assumptions, Limitations & Production Next Steps
- **Assumptions:** Operational queries map to discrete SOP documents; abstaining on low-similarity queries is safer than guessing in high-risk energy operations.
- **Limitations:** Current corpus has 4 short markdown SOPs; TF-IDF lexical matching may miss deep semantic synonyms without dense embeddings.
- **Next Steps:** Integrate dense vector embeddings (`text-embedding-004` / FAISS) with BM25 hybrid reranking, PDF/table parsing for SCADA diagrams, and role-based access control.
