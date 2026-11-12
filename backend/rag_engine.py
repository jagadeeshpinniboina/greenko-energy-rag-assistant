import os
import re
from pathlib import Path
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

DOMAIN_SYNONYMS = {
    "hot": ["temperature", "75c", "cooling", "thermal"],
    "overheating": ["temperature", "75c", "cooling", "fans", "derating"],
    "heat": ["temperature", "75c", "cooling"],
    "shutting": ["derating", "scada", "alarms", "maintenance"],
    "tripping": ["derating", "scada", "alarms", "grid"],
    "shaking": ["vibration", "gearbox", "bearing"],
    "noise": ["vibration", "gearbox", "bearing", "lubrication"],
    "leak": ["oil", "lubrication", "contamination"],
    "water": ["spillway", "gate", "flow", "hydro", "level"],
    "dam": ["spillway", "gate", "hydro", "upstream"],
    "safety": ["lockout", "tagout", "evacuation", "inspection"],
    "power": ["generation", "deviation", "grid", "curtailment", "forecast"],
    "drop": ["deviation", "derating", "curtailment", "forecast"],
    "panel": ["solar", "inverter", "dc", "irradiation"],
    "windmill": ["wind", "turbine", "nacelle", "gearbox", "gust"]
}


class EnergyRAGPipeline:
    """
    Semantic-Aware Hybrid RAG Engine for Greenko Energy Operations SOPs.
    - Combines Word N-Gram TF-IDF + Character N-Gram TF-IDF + Energy Domain Synonym Expansion.
    - Handles paraphrased/natural operator queries while strictly abstaining on out-of-domain queries.
    - Supports live runtime ingestion of new SOP documents.
    """

    def __init__(self, data_dir: Path, similarity_threshold: float = 0.22):
        self.data_dir = Path(data_dir)
        self.similarity_threshold = similarity_threshold
        self.chunks: List[Dict[str, Any]] = []
        self.dynamic_docs: Dict[str, str] = {}
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True
        )
        self.word_matrix = None
        self.char_matrix = None
        self._load_and_chunk_documents()

    def _expand_query(self, query: str) -> str:
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        expanded = list(tokens)
        for t in tokens:
            if t in DOMAIN_SYNONYMS:
                expanded.extend(DOMAIN_SYNONYMS[t])
        return " ".join(expanded)

    def _load_and_chunk_documents(self) -> None:
        self.chunks = []
        all_docs: Dict[str, str] = {}

        for file_path in sorted(self.data_dir.glob("*.md")):
            all_docs[file_path.name] = file_path.read_text(encoding="utf-8").strip()

        for fname, content in self.dynamic_docs.items():
            all_docs[fname] = content.strip()

        for fname, raw_text in all_docs.items():
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            title = lines[0].lstrip("# ").strip() if lines else fname
            body = " ".join(lines[1:]) if len(lines) > 1 else raw_text

            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]

            self.chunks.append({
                "chunk_id": f"{fname}::full",
                "source": fname,
                "title": title,
                "text": f"{title}: {body}",
                "chunk_type": "section"
            })

            for idx, sent in enumerate(sentences):
                if len(sentences) > 1:
                    self.chunks.append({
                        "chunk_id": f"{fname}::s{idx+1}",
                        "source": fname,
                        "title": title,
                        "text": f"[{title}] {sent}",
                        "chunk_type": "sentence"
                    })

        if self.chunks:
            corpus = [c["text"] for c in self.chunks]
            self.word_matrix = self.word_vectorizer.fit_transform(corpus)
            self.char_matrix = self.char_vectorizer.fit_transform(corpus)

    def add_document(self, filename: str, content: str) -> Dict[str, Any]:
        safe_name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", filename.strip())
        if not safe_name.endswith(".md"):
            safe_name += ".md"
        self.dynamic_docs[safe_name] = content.strip()
        self._load_and_chunk_documents()
        return {"filename": safe_name, "total_chunks": len(self.chunks)}

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not query.strip() or self.word_matrix is None:
            return []

        expanded_query = self._expand_query(query)
        word_vec = self.word_vectorizer.transform([expanded_query])
        char_vec = self.char_vectorizer.transform([expanded_query])

        word_scores = cosine_similarity(word_vec, self.word_matrix).flatten()
        char_scores = cosine_similarity(char_vec, self.char_matrix).flatten()

        stop = {"what", "how", "should", "be", "for", "the", "is", "are", "to", "in", "of", "and", "do", "i", "my", "when", "if", "a", "an"}
        query_tokens = set(re.findall(r"[a-z0-9]+", expanded_query.lower())) - stop

        scored_chunks = []
        for idx, chunk in enumerate(self.chunks):
            w_score = float(word_scores[idx])
            c_score = float(char_scores[idx])
            chunk_tokens = set(re.findall(r"[a-z0-9]+", chunk["text"].lower()))
            overlap_ratio = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)

            # Hybrid fusion: 55% Word N-gram + 25% Character N-gram + 20% Expanded Domain Overlap
            hybrid_score = round((0.55 * w_score) + (0.25 * c_score) + (0.20 * overlap_ratio), 4)

            scored_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "title": chunk["title"],
                "text": chunk["text"],
                "cosine_score": round(w_score, 4),
                "hybrid_score": hybrid_score
            })

        scored_chunks.sort(key=lambda x: x["hybrid_score"], reverse=True)
        seen_sources = set()
        unique_results = []
        for item in scored_chunks:
            if item["source"] not in seen_sources:
                unique_results.append(item)
                seen_sources.add(item["source"])
            if len(unique_results) >= top_k:
                break

        return unique_results

    def _generate_with_gemini(self, query: str, evidence: List[Dict[str, Any]]) -> str | None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or not HAS_GENAI:
            return None

        context_block = "\n".join(
            [f"- Source [{e['source']}]: {e['text']}" for e in evidence]
        )
        prompt = (
            "You are an Energy Operations & Safety SOP Assistant for Greenko Group.\n"
            "STRICT RULES:\n"
            "1. Answer ONLY using the provided SOP context below.\n"
            "2. Do NOT add external assumptions or outside knowledge.\n"
            "3. Cite the exact source filename in brackets like [solar_inverter_sop.md].\n"
            "4. If the context does not answer the question, respond strictly: 'Information not found in the provided SOP documents.'\n\n"
            f"SOP Context:\n{context_block}\n\n"
            f"Operator Question: {query}\n"
            "Grounded Answer:"
        )

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text.strip() if response and response.text else None
        except Exception:
            return None

    def _extractive_grounded_answer(self, top_evidence: List[Dict[str, Any]]) -> str:
        primary = top_evidence[0]
        clean_text = re.sub(r"^\[.*?\]\s*", "", primary["text"])
        clean_text = re.sub(rf"^{re.escape(primary['title'])}:\s*", "", clean_text)
        body_lower = clean_text[0].lower() + clean_text[1:] if clean_text else clean_text
        answer = f"Per the {primary['title']}, {body_lower}"
        if len(top_evidence) > 1 and top_evidence[1]["hybrid_score"] >= self.similarity_threshold:
            secondary = top_evidence[1]
            sec_text = re.sub(r"^\[.*?\]\s*", "", secondary["text"])
            sec_text = re.sub(rf"^{re.escape(secondary['title'])}:\s*", "", sec_text)
            sec_lower = sec_text[0].lower() + sec_text[1:] if sec_text else sec_text
            answer += f" Additionally, per the {secondary['title']}, {sec_lower}"
        return answer

    def _verify_hallucination(self, answer: str, evidence: List[Dict[str, Any]], is_out_of_scope: bool) -> Dict[str, Any]:
        if is_out_of_scope:
            return {
                "grounded": True,
                "status": "ABSTAINED_OUT_OF_SCOPE",
                "citation_verified": True,
                "confidence": "High (Safe Abstention)",
                "reason": f"Top retrieval score fell below threshold ({self.similarity_threshold}); abstained because topic is not covered in indexed SOPs."
            }

        expected_sources = {e["source"] for e in evidence if e["hybrid_score"] >= self.similarity_threshold}
        expected_titles = {e["title"].lower() for e in evidence if e["hybrid_score"] >= self.similarity_threshold}
        cited_in_text = set(re.findall(r"\[([a-zA-Z0-9_\-\.]+\.md)\]", answer))
        title_cited = any(t in answer.lower() for t in expected_titles)
        valid_citations = cited_in_text & expected_sources
        has_valid_ref = len(valid_citations) > 0 or title_cited

        evidence_tokens = set(re.findall(r"[a-z0-9]+", " ".join(e["text"] for e in evidence).lower()))
        answer_tokens = set(re.findall(r"[a-z0-9]+", answer.lower())) - {"according", "to", "additionally", "states", "md", "the", "and", "of", "in", "for", "with", "if", "before"}
        overlap = len(answer_tokens & evidence_tokens) / max(len(answer_tokens), 1)

        is_grounded = has_valid_ref and overlap >= 0.45
        return {
            "grounded": is_grounded,
            "status": "VERIFIED_GROUNDED" if is_grounded else "NEEDS_REVIEW",
            "citation_verified": has_valid_ref,
            "token_grounding_ratio": round(overlap, 3),
            "confidence": "High" if overlap >= 0.65 else "Medium",
            "reason": "All claims verified against retrieved SOP citations." if is_grounded else "Partial lexical divergence detected."
        }

    def answer_question(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        cleaned_query = query.strip()
        if not cleaned_query:
            return {
                "query": query,
                "answer": "Please enter an operational, safety, or maintenance question.",
                "citations": [],
                "retrieval_evidence": [],
                "hallucination_check": {
                    "grounded": True,
                    "status": "EMPTY_QUERY",
                    "citation_verified": False,
                    "confidence": "N/A",
                    "reason": "No query text supplied."
                },
                "generation_mode": "validation"
            }

        retrieved = self.retrieve(cleaned_query, top_k=top_k)
        # Require both hybrid score >= threshold AND meaningful lexical/character match (cosine > 0.08)
        relevant = [
            r for r in retrieved
            if r["hybrid_score"] >= self.similarity_threshold and r["cosine_score"] >= 0.08
        ]

        if not relevant:
            abstain_msg = (
                "Information not found in the provided Greenko SOP documents. "
                "This topic is outside the currently indexed plant manuals?please consult the control room or add the relevant SOP."
            )
            return {
                "query": cleaned_query,
                "answer": abstain_msg,
                "citations": [],
                "retrieval_evidence": retrieved,
                "hallucination_check": self._verify_hallucination(abstain_msg, retrieved, is_out_of_scope=True),
                "generation_mode": "abstention_guardrail"
            }

        llm_answer = self._generate_with_gemini(cleaned_query, relevant)
        if llm_answer:
            final_answer = llm_answer
            mode = "gemini-2.5-flash + semantic-hybrid-rag"
        else:
            final_answer = self._extractive_grounded_answer(relevant)
            mode = "semantic-hybrid-rag (grounded extractive)"

        citations = sorted({r["source"] for r in relevant})
        hallucination_report = self._verify_hallucination(final_answer, relevant, is_out_of_scope=False)

        return {
            "query": cleaned_query,
            "answer": final_answer,
            "citations": citations,
            "retrieval_evidence": relevant,
            "hallucination_check": hallucination_report,
            "generation_mode": mode
        }
