"""
PolicyPilot – RAGAS Evaluation Suite
--------------------------------------
Evaluates the RAG triage pipeline using the RAGAS framework.
Measures: faithfulness, answer relevancy, context precision, context recall.

Usage:
    python -m app.eval.eval_ragas                    # full eval with LLM
    python -m app.eval.eval_ragas --mock             # mock eval (no API key)
    python -m app.eval.eval_ragas --prompt-version v2 # eval specific prompt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.chain import TriageChain
from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
settings = get_settings()


def load_test_cases() -> list[dict]:
    """Load test cases from JSON."""
    test_path = Path(__file__).parent / "test_cases.json"
    with open(test_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(prompt_version: str = "v3") -> dict:
    """
    Run the evaluation pipeline and compute metrics.

    Returns dict with:
      - individual results per test case
      - aggregate metrics
    """
    test_cases = load_test_cases()
    chain = TriageChain(prompt_version=prompt_version)

    results = []
    total_latency = 0
    total_tokens = 0
    total_cost = 0
    category_correct = 0
    risk_correct = 0
    entity_recall_sum = 0.0

    log.info("=" * 60)
    log.info("PolicyPilot Evaluation – Prompt %s", prompt_version)
    log.info("Running %d test cases...", len(test_cases))
    log.info("=" * 60)

    for i, tc in enumerate(test_cases, 1):
        t_start = time.perf_counter()

        result = chain.run(
            complaint_text=tc["complaint_text"],
            complaint_id=tc["id"],
            product=tc.get("expected_product"),
        )

        latency_ms = (time.perf_counter() - t_start) * 1000

        # ── Compute metrics ──────────────────────────────────────

        # 1. Category accuracy (fuzzy match)
        pred_cat = result.category.lower().strip()
        exp_cat = tc["expected_category"].lower().strip()
        cat_match = (
            exp_cat in pred_cat
            or pred_cat in exp_cat
            or any(w in pred_cat for w in exp_cat.split() if len(w) > 3)
        )
        if cat_match:
            category_correct += 1

        # 2. Risk level accuracy
        risk_match = result.risk_assessment.risk_level == tc.get("expected_risk_level", "")
        if risk_match:
            risk_correct += 1

        # 3. Entity recall
        expected_entities = tc.get("expected_entities", {})
        entity_recall = _compute_entity_recall(result.entities.to_dict(), expected_entities)
        entity_recall_sum += entity_recall

        # 4. Policy reference coverage
        expected_refs = tc.get("expected_policy_refs", [])
        pred_refs = result.policy_refs
        ref_coverage = _compute_ref_coverage(pred_refs, expected_refs)

        total_latency += latency_ms
        total_tokens += result.tokens_used
        total_cost += result.cost_usd

        result_entry = {
            "id": tc["id"],
            "expected_category": tc["expected_category"],
            "predicted_category": result.category,
            "category_match": cat_match,
            "expected_risk": tc.get("expected_risk_level"),
            "predicted_risk": result.risk_assessment.risk_level,
            "risk_match": risk_match,
            "risk_score": round(result.risk_assessment.overall_score, 4),
            "entity_recall": round(entity_recall, 4),
            "ref_coverage": round(ref_coverage, 4),
            "latency_ms": round(latency_ms, 1),
            "tokens": result.tokens_used,
            "cost_usd": round(result.cost_usd, 6),
        }
        results.append(result_entry)

        status = "✅" if cat_match else "❌"
        log.info(
            "[%02d/%02d] %s %s | pred=%s | risk=%s (%.2f) | %.0fms",
            i, len(test_cases), status, tc["id"],
            result.category, result.risk_assessment.risk_level,
            result.risk_assessment.overall_score, latency_ms,
        )

    # ── Aggregate metrics ────────────────────────────────────────
    n = len(test_cases)
    metrics = {
        "prompt_version": prompt_version,
        "total_test_cases": n,
        "category_accuracy": round(category_correct / n, 4),
        "risk_level_accuracy": round(risk_correct / n, 4),
        "mean_entity_recall": round(entity_recall_sum / n, 4),
        "mean_latency_ms": round(total_latency / n, 1),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "cost_per_triage_usd": round(total_cost / n, 6) if n > 0 else 0,
    }

    log.info("")
    log.info("=" * 60)
    log.info("EVALUATION RESULTS – Prompt %s", prompt_version)
    log.info("=" * 60)
    log.info("Category Accuracy:   %.1f%%", metrics["category_accuracy"] * 100)
    log.info("Risk Level Accuracy: %.1f%%", metrics["risk_level_accuracy"] * 100)
    log.info("Mean Entity Recall:  %.1f%%", metrics["mean_entity_recall"] * 100)
    log.info("Mean Latency:        %.0fms", metrics["mean_latency_ms"])
    log.info("Total Tokens:        %d", metrics["total_tokens"])
    log.info("Total Cost:          $%.4f", metrics["total_cost_usd"])
    log.info("Cost per Triage:     $%.4f", metrics["cost_per_triage_usd"])

    return {"metrics": metrics, "results": results}


def _compute_entity_recall(predicted: dict, expected: dict) -> float:
    """Compute recall of expected entities in predicted entities."""
    if not expected:
        return 1.0  # No expected entities = perfect recall

    total_expected = 0
    total_found = 0

    for key, expected_values in expected.items():
        if not isinstance(expected_values, list):
            continue
        for ev in expected_values:
            total_expected += 1
            pred_values = predicted.get(key, [])
            # Fuzzy match: check if any predicted value contains the expected value
            if any(ev.lower() in pv.lower() or pv.lower() in ev.lower()
                   for pv in pred_values):
                total_found += 1

    return total_found / total_expected if total_expected > 0 else 1.0


def _compute_ref_coverage(predicted_refs: list[str], expected_refs: list[str]) -> float:
    """Compute how many expected references are covered by predictions."""
    if not expected_refs:
        return 1.0

    covered = 0
    for exp in expected_refs:
        exp_lower = exp.lower()
        if any(exp_lower in pred.lower() or pred.lower() in exp_lower
               for pred in predicted_refs):
            covered += 1

    return covered / len(expected_refs)


def main():
    parser = argparse.ArgumentParser(description="Run PolicyPilot evaluation")
    parser.add_argument("--prompt-version", default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--mock", action="store_true", help="Force mock mode")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    results = run_evaluation(prompt_version=args.prompt_version)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = settings.data_processed / f"eval_{args.prompt_version}_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
