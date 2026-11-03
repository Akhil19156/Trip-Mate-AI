import json
from pathlib import Path

from backend.evaluations.run_eval import evaluate_case, run_evaluation


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DATASET = ROOT / "backend" / "evaluations" / "cases.json"


def test_offline_evaluation_dataset_passes_with_project_validator():
    import backend.graph as project_backend

    report = run_evaluation(EVALUATION_DATASET, project_backend)
    assert report["case_count"] == 3
    assert report["case_pass_rate"] == 1
    assert report["constraint_accuracy"] == 1
    assert report["budget_math_accuracy"] == 1
    assert report["itinerary_validity_rate"] == 1


def test_offline_evaluation_detects_wrong_budget_total():
    import backend.graph as project_backend

    case = json.loads(
        EVALUATION_DATASET.read_text(encoding="utf-8")
    )["cases"][0]
    case["recorded"]["budget_total"] = 4000
    result = evaluate_case(case, project_backend)
    assert not result["budget_math_correct"]
    assert not result["passed"]


def test_offline_evaluation_detects_unsupported_claim():
    import backend.graph as project_backend

    case = json.loads(
        EVALUATION_DATASET.read_text(encoding="utf-8")
    )["cases"][0]
    case["recorded"]["claims"] = ["A direct train arrives every hour."]
    result = evaluate_case(case, project_backend)
    assert result["grounded_claims"] == 0
    assert not result["passed"]


def _valid_live_trip():
    return {
        "user_query": "Plan a 3 day trip to Rishikesh under 4000 INR",
        "guardrail_allowed": True,
        "trip_constraints": {
            "destination": "Rishikesh",
            "duration_days": 3,
            "budget": "under 4000 INR",
            "destination_iata": "",
        },
        "selected_agents": [
            "hotel_agent",
            "weather_agent",
            "budget_agent",
            "itinerary_agent",
        ],
        "day_by_day_plan": (
            "## Day-by-day plan\n"
            "### Day 1 - Arrival\n- Settle in.\n"
            "### Day 2 - Explore\n- Explore the area.\n"
            "### Day 3 - Return\n- Travel home."
        ),
        "itinerary": (
            "## Trip itinerary\n"
            "## Getting there\nRishikesh has no commercial airport. "
            "Nearest: Dehradun (DED) for Rishikesh."
        ),
        "budget_breakdown": {
            "transport": 800,
            "stay": 1800,
            "food": 600,
            "activities": 300,
            "local_travel": 250,
            "buffer": 200,
        },
        "budget_total": 3950,
        "budget_feasible": True,
        "budget_gap": 50,
        "hotel_data": {"data_type": "unavailable"},
        "weather_data": {"data_type": "unavailable"},
        "messages": [],
    }


def test_live_evaluation_scores_actual_serialized_response():
    import backend.graph as project_backend

    response = project_backend._serialize_result(_valid_live_trip(), "live-thread")
    assert response["evaluation"]["type"] == "deterministic_live_checks"
    assert response["evaluation"]["passed"]
    assert response["evaluation"]["score"] == 100


def test_live_evaluation_flags_wrong_budget_feasibility():
    import backend.graph as project_backend

    state = _valid_live_trip()
    state["budget_feasible"] = False
    response = project_backend._serialize_result(state, "live-thread")
    check = next(
        check
        for check in response["evaluation"]["checks"]
        if check["name"] == "budget_feasibility"
    )
    assert check["status"] == "failed"
    assert not response["evaluation"]["passed"]
