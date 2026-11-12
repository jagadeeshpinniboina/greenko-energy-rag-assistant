import json
from pathlib import Path
import pandas as pd
from backend.rag_engine import EnergyRAGPipeline

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"


def run_evaluation():
    rag = EnergyRAGPipeline(data_dir=DATA, similarity_threshold=0.29)
    df = pd.read_csv(DATA / "questions.csv")

    eval_cases = [
        {
            "question": row["question"],
            "expected_source": row["expected_source"],
            "type": "in_scope_benchmark"
        }
        for _, row in df.iterrows()
    ] + [
        {
            "question": "What checks are needed before spillway gate inspection?",
            "expected_source": "hydro_safety_brief.md",
            "type": "in_scope_extended"
        },
        {
            "question": "What should be reviewed before escalating generation deviation?",
            "expected_source": "grid_operations_note.md",
            "type": "in_scope_extended"
        },
        {
            "question": "My solar inverter is overheating and shutting down, what should I inspect?",
            "expected_source": "solar_inverter_sop.md",
            "type": "in_scope_paraphrased_natural"
        },
        {
            "question": "The windmill gearbox is shaking and leaking oil, how do we check it?",
            "expected_source": "wind_turbine_maintenance.md",
            "type": "in_scope_paraphrased_natural"
        },
        {
            "question": "What is the nuclear reactor coolant temperature threshold?",
            "expected_source": None,
            "type": "out_of_scope_hallucination_test"
        },
        {
            "question": "How do I configure a Kubernetes ingress controller?",
            "expected_source": None,
            "type": "out_of_scope_hallucination_test"
        }
    ]

    results = []
    in_scope_total = 0
    top1_hits = 0
    citation_hits = 0
    out_of_scope_total = 0
    abstention_hits = 0

    for case in eval_cases:
        out = rag.answer_question(case["question"], top_k=2)
        top_src = out["retrieval_evidence"][0]["source"] if out["retrieval_evidence"] else None
        cited = out["citations"]
        hc_status = out["hallucination_check"]["status"]

        if case["expected_source"] is not None:
            in_scope_total += 1
            hit_top1 = (top_src == case["expected_source"])
            hit_cite = (case["expected_source"] in cited)
            top1_hits += int(hit_top1)
            citation_hits += int(hit_cite)
            passed = hit_top1 and hit_cite
        else:
            out_of_scope_total += 1
            passed = (hc_status == "ABSTAINED_OUT_OF_SCOPE" and len(cited) == 0)
            abstention_hits += int(passed)

        results.append({
            "question": case["question"],
            "type": case["type"],
            "expected_source": case["expected_source"],
            "predicted_top_source": top_src,
            "citations": cited,
            "hallucination_status": hc_status,
            "passed": passed
        })

    summary = {
        "total_test_cases": len(eval_cases),
        "retrieval_recall_at_1": round(top1_hits / max(in_scope_total, 1), 4),
        "citation_precision_at_k": round(citation_hits / max(in_scope_total, 1), 4),
        "hallucination_abstention_rate": round(abstention_hits / max(out_of_scope_total, 1), 4),
        "overall_pass_rate": round(sum(r["passed"] for r in results) / len(results), 4),
        "detailed_results": results
    }

    print(json.dumps(summary, indent=2))
    (BASE / "evaluation_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_evaluation()
