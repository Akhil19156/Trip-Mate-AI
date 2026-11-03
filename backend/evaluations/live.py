from __future__ import annotations

import re
from typing import Any, Callable


_PLANNING_QUERY = r"\b(?:plan|trip|itinerary)\b|\bdays in\b"
_BUDGET_QUERY = r"\b(?:plan|trip|itinerary|budget|under)\b"
_BUDGET_CATEGORIES = (
    "transport",
    "stay",
    "food",
    "activities",
    "local_travel",
    "buffer",
)


def evaluate_live_result(
    state: dict[str, Any],
    *,
    parse_amount: Callable[[Any], float | None],
    validate_itinerary: Callable[[dict[str, Any]], list[str]],
    has_airport: Callable[[Any], bool],
) -> dict[str, Any]:
    """Score deterministic workflow contracts for one live graph result."""
    checks: list[dict[str, str]] = []

    def add(name: str, label: str, passed: bool | None, detail: str) -> None:
        checks.append(
            {
                "name": name,
                "label": label,
                "status": (
                    "not_applicable"
                    if passed is None
                    else "passed"
                    if passed
                    else "failed"
                ),
                "detail": detail,
            }
        )

    constraints = state.get("trip_constraints", {})
    query = str(state.get("user_query", ""))
    allowed = state.get("guardrail_allowed", True) is True
    destination = str(constraints.get("destination") or "").strip()
    add(
        "guardrail_destination",
        "Request and destination check",
        bool(destination) if allowed else bool(state.get("guardrail_reason")),
        (
            "A destination was extracted."
            if allowed and destination
            else "The request was blocked or needs clarification."
            if not allowed and state.get("guardrail_reason")
            else "An allowed request has no destination."
        ),
    )

    if allowed:
        selected = set(state.get("selected_agents", []))
        planning = bool(re.search(_PLANNING_QUERY, query, re.IGNORECASE))
        budget_required = bool(
            str(constraints.get("budget") or "").strip()
            or re.search(_BUDGET_QUERY, query, re.IGNORECASE)
            or planning
        )
        required_agents = {"itinerary_agent"}
        if planning:
            required_agents.update({"hotel_agent", "weather_agent"})
        if budget_required:
            required_agents.add("budget_agent")
        missing_agents = sorted(required_agents - selected)
        add(
            "required_agent_selection",
            "Required agent selection",
            not missing_agents,
            (
                "All required agents were selected."
                if not missing_agents
                else "Missing: " + ", ".join(missing_agents)
            ),
        )

        destination_iata = constraints.get("destination_iata")
        airport_exists = has_airport(destination_iata)
        flight_selected = "flight_agent" in selected
        add(
            "airport_flight_routing",
            "Airport and flight routing",
            airport_exists or not flight_selected,
            (
                "Flight agent selection matches airport availability."
                if airport_exists or not flight_selected
                else "Flight agent was selected although no destination airport is available."
            ),
        )
        if not airport_exists:
            itinerary = str(state.get("itinerary") or "")
            getting_there_present = "## Getting there" in itinerary
            misleading_flight_error = "service could not be reached" in itinerary.casefold()
            add(
                "ground_transport_display",
                "Ground transport guidance",
                getting_there_present and not misleading_flight_error,
                (
                    "Itinerary explains ground transport and does not imply a failed flight service."
                    if getting_there_present and not misleading_flight_error
                    else "Expected a Getting there section without a flight-service failure message."
                ),
            )

        if state.get("day_by_day_plan"):
            itinerary_problems = validate_itinerary(state)
            add(
                "itinerary_structure",
                "Itinerary structure",
                not itinerary_problems,
                (
                    "Required sections and day count are valid."
                    if not itinerary_problems
                    else "; ".join(itinerary_problems)
                ),
            )
        else:
            add(
                "itinerary_structure",
                "Itinerary structure",
                None,
                "No itinerary draft was produced for this response.",
            )

        if "budget_agent" in selected:
            budget = state.get("budget_breakdown", {})
            amounts = [parse_amount(budget.get(key)) for key in _BUDGET_CATEGORIES]
            total = parse_amount(state.get("budget_total"))
            math_ok = (
                all(amount is not None and amount > 0 for amount in amounts)
                and total is not None
                and round(sum(amounts), 2) == round(total, 2)
            )
            add(
                "budget_math",
                "Budget arithmetic",
                math_ok,
                (
                    "All category values are positive and sum to the recorded total."
                    if math_ok
                    else "Budget categories are missing, non-positive, or do not sum to the total."
                ),
            )
            user_budget = parse_amount(constraints.get("budget"))
            feasible = state.get("budget_feasible")
            gap = parse_amount(state.get("budget_gap"))
            if user_budget is None:
                add(
                    "budget_feasibility",
                    "Budget feasibility",
                    None,
                    "No numeric user budget was available to compare.",
                )
            elif total is None:
                add(
                    "budget_feasibility",
                    "Budget feasibility",
                    False,
                    "A stated budget exists but no computed total is available.",
                )
            else:
                feasibility_ok = (
                    feasible is (total <= user_budget)
                    and gap is not None
                    and round(gap, 2) == round(user_budget - total, 2)
                )
                add(
                    "budget_feasibility",
                    "Budget feasibility",
                    feasibility_ok,
                    (
                        "Feasibility and budget gap match the computed total."
                        if feasibility_ok
                        else "Feasibility or budget gap disagrees with the computed total."
                    ),
                )
        else:
            add(
                "budget_math",
                "Budget arithmetic",
                None,
                "Budget agent was not selected for this request.",
            )

        for agent_name, state_key, label in (
            ("hotel_agent", "hotel_data", "Hotel data status"),
            ("weather_agent", "weather_data", "Weather data status"),
        ):
            if agent_name not in selected:
                add(
                    f"{agent_name}_status",
                    label,
                    None,
                    "Agent was not selected.",
                )
                continue
            tool_data = state.get(state_key, {})
            data_type = (
                tool_data.get("data_type")
                if isinstance(tool_data, dict)
                else None
            )
            add(
                f"{agent_name}_status",
                label,
                data_type in {"live", "estimate", "unavailable"},
                (
                    f"Reported status: {data_type}."
                    if data_type in {"live", "estimate", "unavailable"}
                    else "Tool result has no recognized availability status."
                ),
            )

    applicable = [check for check in checks if check["status"] != "not_applicable"]
    passed_count = sum(check["status"] == "passed" for check in applicable)
    score = round(100 * passed_count / len(applicable)) if applicable else None
    failures = sum(check["status"] == "failed" for check in applicable)
    return {
        "type": "deterministic_live_checks",
        "passed": failures == 0,
        "score": score,
        "passed_checks": passed_count,
        "total_checks": len(applicable),
        "checks": checks,
        "note": (
            "Checks workflow contracts and output structure; it does not judge "
            "subjective itinerary quality or verify facts independently."
        ),
    }
