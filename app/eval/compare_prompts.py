"""
PolicyPilot – Prompt Version Comparison
-----------------------------------------
Runs the evaluation suite across all prompt versions (v1, v2, v3)
and generates a comparison report with visualisation.

Usage:
    python -m app.eval.compare_prompts
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import get_settings
from app.eval.eval_ragas import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
settings = get_settings()

VERSIONS = ["v1", "v2", "v3"]


def compare_prompts() -> dict:
    """Run evaluation for all prompt versions and compare."""
    all_results = {}

    for version in VERSIONS:
        log.info("\n" + "=" * 60)
        log.info("EVALUATING PROMPT VERSION: %s", version)
        log.info("=" * 60)

        result = run_evaluation(prompt_version=version)
        all_results[version] = result

        # Save individual results
        output_path = settings.data_processed / f"eval_{version}_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # ── Build comparison report ──────────────────────────────────
    comparison = {
        "versions": {},
        "winner": None,
        "recommendation": "",
    }

    log.info("\n" + "=" * 60)
    log.info("PROMPT COMPARISON REPORT")
    log.info("=" * 60)
    log.info("")
    log.info("%-12s %-12s %-12s %-12s %-12s %-10s",
             "Version", "Cat.Acc", "Risk.Acc", "Ent.Recall", "Latency", "Cost/req")
    log.info("-" * 72)

    best_score = -1
    best_version = "v3"

    for version in VERSIONS:
        m = all_results[version]["metrics"]
        comparison["versions"][version] = m

        # Composite score: weighted sum of accuracy metrics
        composite = (
            m["category_accuracy"] * 0.4
            + m["risk_level_accuracy"] * 0.3
            + m["mean_entity_recall"] * 0.3
        )

        log.info(
            "%-12s %-12s %-12s %-12s %-12s $%-10s",
            version,
            f'{m["category_accuracy"]:.1%}',
            f'{m["risk_level_accuracy"]:.1%}',
            f'{m["mean_entity_recall"]:.1%}',
            f'{m["mean_latency_ms"]:.0f}ms',
            f'{m["cost_per_triage_usd"]:.4f}',
        )

        if composite > best_score:
            best_score = composite
            best_version = version

    comparison["winner"] = best_version
    comparison["recommendation"] = (
        f"Prompt {best_version} achieves the best composite score. "
        f"Recommended for production use."
    )

    log.info("")
    log.info("🏆 Winner: Prompt %s", best_version)
    log.info("   %s", comparison["recommendation"])

    # Save comparison report
    report_path = settings.data_processed / "prompt_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    log.info("Report saved to %s", report_path)

    return comparison


def main():
    compare_prompts()


if __name__ == "__main__":
    main()
