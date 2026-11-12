import os
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.rag_engine import EnergyRAGPipeline

app = FastAPI(
    title="Energy Operations Document RAG Assistant",
    description="Grounded RAG assistant for Greenko renewable energy operations, safety, and maintenance SOPs.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
FRONTEND = BASE / "frontend"

rag_pipeline = EnergyRAGPipeline(data_dir=DATA, similarity_threshold=0.29)


class Query(BaseModel):
    text: str = Field(..., description="Operational or maintenance question from engineer/operator")
    top_k: int = Field(default=2, ge=1, le=5)


@app.get("/")
def serve_frontend():
    index_path = FRONTEND / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"status": "Backend running. Open /docs for API playground."}


@app.get("/api/demo")
def demo():
    questions_file = DATA / "questions.csv"
    sample_questions = []
    if questions_file.exists():
        df = pd.read_csv(questions_file)
        sample_questions = df.to_dict(orient="records")

    return {
        "case_study": "Energy Operations Document RAG Assistant",
        "status": "healthy",
        "indexed_documents": sorted([p.name for p in DATA.glob("*.md")]),
        "total_chunks": len(rag_pipeline.chunks),
        "sample_questions": sample_questions,
        "expected_outcome": "A working RAG-based assistant that answers operational questions from provided documents with citations, retrieval evidence, and hallucination checks."
    }


@app.get("/api/documents")
def list_documents():
    docs = []
    for md_file in sorted(DATA.glob("*.md")):
        docs.append({
            "filename": md_file.name,
            "content": md_file.read_text(encoding="utf-8").strip()
        })
    for fname, content in rag_pipeline.dynamic_docs.items():
        docs.append({
            "filename": fname,
            "content": content.strip()
        })
    return {"documents": docs}



class NewSOP(BaseModel):
    filename: str
    content: str


@app.post("/api/documents/add")
def add_sop_document(doc: NewSOP):
    info = rag_pipeline.add_document(doc.filename, doc.content)
    return {"status": "indexed", **info}


@app.post("/api/run")
def run_case(query: Query):
    result = rag_pipeline.answer_question(query.text, top_k=query.top_k)
    return {
        "input": result["query"],
        "answer": result["answer"],
        "citations": result["citations"],
        "retrieval_evidence": result["retrieval_evidence"],
        "hallucination_check": result["hallucination_check"],
        "generation_mode": result["generation_mode"],
        "limitations": [
            "Knowledge base currently covers 4 bundled SOP markdown files.",
            "Threshold-based abstention prioritizes zero hallucination over answering ambiguous out-of-scope queries."
        ]
    }
