from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _load_backend():
    os.environ.setdefault("DATABASE_URL", "postgresql://offline:offline@localhost/offline")
    os.environ.setdefault("GROQ_API_KEY", "offline-evaluation")
    sys.path.insert(0, str(ROOT))

    import psycopg_pool
    from langgraph.checkpoint.postgres import PostgresSaver

    class OfflineConnectionPool:
        check_connection = staticmethod(lambda connection: None)

        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    psycopg_pool.ConnectionPool = OfflineConnectionPool
    PostgresSaver.setup = lambda self: None
    return importlib.import_module("backend.graph")


def _same_constraint(name: str, expected: Any, actual: Any, backend) -> bool:
    if name == "destination":
        return str(expected or "").strip().casefold() == str(actual or "").strip().casefold()
    if name == "duration_days":
        return expected == actual
    if name == "budget":
        expected_amount = backend.parse_amount(expected)
        actual_amount = backend.parse_amount(actual)
        if expected_amount is not None or actual_amount is not None:
            return expected_amount == actual_amount
        return str(expected or "").strip().casefold() == str(actual or "").strip().casefold()
    raise ValueError(f"Unsupported constraint in evaluation dataset: {name}")


def evaluate_case(case: dict[str, Any], backend) -> dict[str, Any]:
    expected = case["expected"]
    recorded = case["recorded"]
    expected_constraints = expected.get("constraints", {})
    recorded_constraints = recorded.get("trip_constraints", {})

    constraint_results = {
        name: _same_constraint(name, value, recorded_constraints.get(name), backend)
        for name, value in expected_constraints.items()
    }
    expected_agents = set(expected.get("selected_agents", []))
    recorded_agents = set(recorded.get("selected_agents", []))
    agent_selection_correct = expected_agents == recorded_agents

    budget = recorded.get("budget_breakdown", {})
    category_names = (
        "transport",
        "stay",
        "food",
        "activities",
        "local_travel",
        "buffer",
    )
    categories = [backend.parse_amount(budget.get(name)) for name in category_names]
    recorded_total = backend.parse_amount(recorded.get("budget_total"))
    budget_math_correct = (
        all(value is not None for value in categories)
        and recorded_total is not None
        and round(sum(categories), 2) == round(recorded_total, 2)
    )

    validation_problems = backend.validate_itinerary(
        {
            "day_by_day_plan": recorded.get("day_by_day_plan", ""),
            "trip_constraints": {
                "duration_days": expected_constraints.get("duration_days")
            },
        }
    )

    source_facts = [
        str(fact).strip().casefold()
        for fact in recorded.get("source_facts", [])
    ]
    claims = recorded.get("claims", [])
    grounded_claims = [
        any(str(claim).strip().casefold() in source for source in source_facts)
        for claim in claims
    ]
    grounding_correct = all(grounded_claims)

    checks = [
        *constraint_results.values(),
        agent_selection_correct,
        budget_math_correct,
        not validation_problems,
        grounding_correct,
    ]
    return {
        "id": case["id"],
        "constraint_results": constraint_results,
        "agent_selection_correct": agent_selection_correct,
        "budget_math_correct": budget_math_correct,
        "itinerary_valid": not validation_problems,
        "validation_problems": validation_problems,
        "grounded_claims": len([item for item in grounded_claims if item]),
        "total_claims": len(claims),
        "passed": all(checks),
    }


def run_evaluation(dataset_path: Path, backend=None) -> dict[str, Any]:
    if backend is None:
        backend = _load_backend()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must contain a non-empty cases list.")

    results = [evaluate_case(case, backend) for case in cases]
    constraint_checks = [
        passed
        for result in results
        for passed in result["constraint_results"].values()
    ]
    total_claims = sum(result["total_claims"] for result in results)
    grounded_claims = sum(result["grounded_claims"] for result in results)
    summary = {
        "dataset": str(dataset_path),
        "case_count": len(results),
        "case_pass_rate": sum(result["passed"] for result in results) / len(results),
        "constraint_accuracy": (
            sum(constraint_checks) / len(constraint_checks)
            if constraint_checks
            else None
        ),
        "agent_selection_accuracy": sum(
            result["agent_selection_correct"] for result in results
        ) / len(results),
        "budget_math_accuracy": sum(
            result["budget_math_correct"] for result in results
        ) / len(results),
        "itinerary_validity_rate": sum(
            result["itinerary_valid"] for result in results
        ) / len(results),
        "grounded_claim_rate": (
            grounded_claims / total_claims if total_claims else None
        ),
        "results": results,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score saved TripMate outputs without making LLM or MCP calls."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "backend" / "evaluations" / "cases.json",
        help="Path to a JSON dataset of expected labels and recorded agent outputs.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to save the complete JSON evaluation report.",
    )
    args = parser.parse_args()

    report = run_evaluation(args.dataset)
    print(f"Offline evaluation: {report['case_count']} saved cases")
    for metric in (
        "case_pass_rate",
        "constraint_accuracy",
        "agent_selection_accuracy",
        "budget_math_accuracy",
        "itinerary_validity_rate",
        "grounded_claim_rate",
    ):
        value = report[metric]
        formatted = "n/a" if value is None else f"{value:.0%}"
        print(f"{metric}: {formatted}")
    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {result['id']}")
        for problem in result["validation_problems"]:
            print(f"  - {problem}")
        for name, passed in result["constraint_results"].items():
            if not passed:
                print(f"  - constraint mismatch: {name}")
        for check in (
            "agent_selection_correct",
            "budget_math_correct",
            "itinerary_valid",
        ):
            if not result[check]:
                print(f"  - failed check: {check}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0 if all(result["passed"] for result in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
