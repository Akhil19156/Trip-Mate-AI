import pytest
import uuid
from types import SimpleNamespace
from collections import deque
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from fastapi.testclient import TestClient

import backend.app as api
import backend.graph as backend
from backend.auth import require_csrf
from backend.graph import (
    TravelState,
    budget_agent,
    extract_destination,
    final_agent,
    human_approval_agent,
    itinerary_agent,
    parse_amount,
    revise_itinerary,
    render_budget_section,
    render_flight_status,
    render_weather_section,
    route_after_review,
    route_after_itinerary,
    supervisor_agent,
    validate_itinerary,
)
from backend.schemas import SupervisorPlan, TripBudget, TripConstraints


@pytest.fixture(autouse=True)
def clear_token_window():
    backend._token_history.clear()
    yield
    backend._token_history.clear()


@pytest.fixture
def authenticated_client(monkeypatch):
    user = SimpleNamespace(id=uuid.uuid4(), preferences=None)
    monkeypatch.setitem(api.app.dependency_overrides, require_csrf, lambda: user)
    monkeypatch.setattr(api, "preference_context", lambda _user: {})
    return TestClient(api.app)


def _state(**overrides):
    state = {
        "user_query": "Plan a 2-day trip to Rome",
        "trip_constraints": {
            "destination": "Rome",
            "origin": "",
            "duration_days": 2,
            "budget": "",
            "travel_style": "",
            "special_preferences": [],
        },
        "selected_agents": [],
        "llm_calls": 0,
        "messages": [],
        "budget_total": 10.0,
        "budget_breakdown": {
            "transport": 1.0,
            "stay": 2.0,
            "food": 1.0,
            "activities": 2.0,
            "local_travel": 2.0,
            "buffer": 2.0,
            "currency": "EUR",
        },
        "budget_gap": None,
        "budget_results": "",
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "flight_data": {},
        "hotel_data": {},
        "weather_data": {},
        "itinerary": "",
        "revision_count": 0,
        "max_revisions": 3,
        "approval_status": "pending",
        "approved": False,
        "human_feedback": "",
        "thread_id": "test-thread",
    }
    state.update(overrides)
    return state


def test_guardrail_fails_closed_when_llm_raises(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(backend, "call_llm", fail)
    result = supervisor_agent(_state())
    assert result["guardrail_allowed"] is False
    assert result["selected_agents"] == []
    assert "retry" in result["guardrail_reason"].lower()


@pytest.mark.parametrize("query", ["", " " * 3, "x" * 2001])
def test_guardrail_precheck_does_not_call_llm(monkeypatch, query):
    def fail(*args, **kwargs):
        pytest.fail("The LLM must not run for invalid input.")

    monkeypatch.setattr(backend, "call_llm", fail)
    result = supervisor_agent(_state(user_query=query))
    assert result["guardrail_allowed"] is False
    assert result["llm_calls"] == 0


def test_guardrail_fails_closed_on_invalid_structured_output(monkeypatch):
    monkeypatch.setattr(
        backend,
        "call_llm",
        lambda *args, **kwargs: {
            "allowed": "true",
            "reason": "",
            "selected_agents": ["flight_agent"],
            "trip_constraints": {"destination": "Rome"},
            "reasoning": "",
        },
    )
    result = supervisor_agent(_state())
    assert result["guardrail_allowed"] is False
    assert result["selected_agents"] == []
    assert result["llm_calls"] == 0


def test_supervisor_plan_requires_real_bool():
    with pytest.raises(ValueError):
        SupervisorPlan.model_validate(
            {
                "allowed": "true",
                "reason": "",
                "selected_agents": ["itinerary_agent"],
                "trip_constraints": {"destination": "Rome"},
                "reasoning": "",
            }
        )


def test_supervisor_includes_itinerary_without_selecting_all_on_success(monkeypatch):
    plan = SupervisorPlan(
        allowed=True,
        reason="Travel request",
        selected_agents=["weather_agent"],
        trip_constraints=TripConstraints(destination="Rome"),
        reasoning="Weather requested",
    )
    monkeypatch.setattr(backend, "call_llm", lambda *args, **kwargs: plan)
    result = supervisor_agent(_state())
    assert result["guardrail_allowed"] is True
    assert result["selected_agents"] == [
        "flight_agent",
        "hotel_agent",
        "weather_agent",
        "budget_agent",
        "itinerary_agent",
    ]


def test_supervisor_requests_destination_when_missing(monkeypatch):
    plan = SupervisorPlan(
        allowed=True,
        reason="Travel request",
        selected_agents=["hotel_agent"],
        trip_constraints=TripConstraints(),
        reasoning="Destination is missing",
    )
    monkeypatch.setattr(backend, "call_llm", lambda *args, **kwargs: plan)
    result = supervisor_agent(_state())
    assert result["guardrail_allowed"] is False
    assert result["selected_agents"] == []
    assert "destination" in result["final_response"].lower()


def test_destination_extraction_uses_constraint_before_llm():
    assert extract_destination(
        "some travel request",
        {"destination": "Paris"},
        llm_call=lambda messages: pytest.fail("The LLM must not be called."),
    ) == "Paris"


def test_failed_tool_result_is_marked_unavailable():
    assert backend._tool_result(
        "fixture",
        "unavailable",
        {},
        error="not available",
    )["data_type"] == "unavailable"


def test_hotel_search_uses_supervisor_destination(monkeypatch):
    queries = []

    def fake_search(query):
        queries.append(query)

        async def result():
            return {"results": [{"title": "Sample stay", "url": "https://example.com"}]}

        return result()

    monkeypatch.setattr(backend, "tavily_mcp_search", fake_search)

    def run(coroutine):
        coroutine.close()
        return {"results": [{"title": "Sample stay", "url": "https://example.com"}]}

    monkeypatch.setattr(backend.asyncio, "run", run)
    backend.hotel_agent(
        _state(
            user_query="Find a trip to Rome and discuss unrelated topics",
            trip_constraints={"destination": "Rome"},
        )
    )
    assert queries == [
        "budget and mid-range hotels in Rome price per night INR",
        "top attractions in Rome",
    ]


def test_budget_total_is_computed_in_python(monkeypatch):
    budget = TripBudget(
        transport=100,
        stay=200,
        food=50,
        activities=25,
        local_travel=15,
        buffer=10,
        currency="EUR",
        assumptions=["Estimate only"],
    )
    responses = iter([budget])
    monkeypatch.setattr(
        backend,
        "call_llm",
        lambda *args, **kwargs: next(responses),
    )
    result = budget_agent(_state())
    assert result["budget_total"] == 400.0
    assert result["llm_calls"] == 1
    assert "## Budget" in result["budget_results"]


def test_unwrap_mcp_failure_marks_tool_unavailable(monkeypatch):
    raw = [{"type": "text", "text": '{"ok": false, "error": "service could not be reached"}'}]

    def fake_search(query):
        async def result():
            return raw

        return result()

    monkeypatch.setattr(backend, "tavily_mcp_search", fake_search)
    monkeypatch.setattr(
        backend.asyncio,
        "run",
        lambda coroutine: (coroutine.close(), raw)[1],
    )
    result = backend.hotel_agent(_state())
    assert backend.unwrap_mcp_result(raw) == {
        "ok": False,
        "error": "service could not be reached",
    }
    assert result["hotel_data"]["data_type"] == "unavailable"
    assert "service could not be reached" in result["hotel_results"]


def test_supervisor_always_adds_weather_for_trip_planning(monkeypatch):
    plan = SupervisorPlan(
        allowed=True,
        reason="Travel request",
        selected_agents=["itinerary_agent"],
        trip_constraints=TripConstraints(destination="Rome"),
        reasoning="Trip request",
    )
    monkeypatch.setattr(backend, "call_llm", lambda *args, **kwargs: plan)
    result = supervisor_agent(_state(user_query="Plan a trip to Rome"))
    assert "weather_agent" in result["selected_agents"]
    assert "hotel_agent" in result["selected_agents"]


def test_planning_query_with_budget_selects_budget_agent(monkeypatch):
    plan = SupervisorPlan(
        allowed=True,
        reason="Travel request",
        selected_agents=["itinerary_agent"],
        trip_constraints=TripConstraints(
            destination="Rishikesh",
            budget="under 4000 INR",
        ),
        reasoning="Budgeted trip",
    )
    monkeypatch.setattr(backend, "call_llm", lambda *args, **kwargs: plan)
    result = supervisor_agent(
        _state(user_query="Plan a 3 day trip to rishikesh from Delhi under 4000 INR")
    )
    assert "budget_agent" in result["selected_agents"]


def test_supervisor_validates_airport_codes_and_uses_origin_fallback(monkeypatch):
    plan = SupervisorPlan(
        allowed=True,
        reason="Travel request",
        selected_agents=["itinerary_agent"],
        trip_constraints=TripConstraints(destination="Rome"),
        origin_iata="not-an-airport",
        destination_iata="FCO",
        reasoning="Trip request",
    )
    monkeypatch.setattr(backend, "call_llm", lambda *args, **kwargs: plan)
    result = supervisor_agent(_state(user_query="Plan a trip to Rome"))
    assert result["trip_constraints"]["origin_iata"] == backend.DEFAULT_ORIGIN_IATA
    assert result["trip_constraints"]["destination_iata"] == "FCO"


def test_weather_renderer_distinguishes_not_requested_and_unavailable():
    assert "Weather was not requested" in render_weather_section({}, requested=False)
    unavailable = render_weather_section(
        {"data_type": "unavailable", "error": "service could not be reached"}
    )
    assert "Forecast unavailable (service could not be reached)" in unavailable


def test_tavily_results_with_null_metadata_are_live(monkeypatch):
    payload = {
        "results": [{"title": "Budget stay", "url": "https://booking.com/hotel"}],
        "answer": None,
        "follow_up_questions": None,
        "response_time": 0,
    }

    def fake_search(query):
        async def result():
            return payload

        return result()

    monkeypatch.setattr(backend, "tavily_mcp_search", fake_search)
    monkeypatch.setattr(
        backend.asyncio,
        "run",
        lambda coroutine: (coroutine.close(), payload)[1],
    )
    result = backend.hotel_agent(_state())
    assert result["hotel_data"]["data_type"] == "live"
    assert result["attractions_data"]["data_type"] == "live"


def test_no_airport_removes_flights_and_renders_ground_transport(monkeypatch):
    plan = SupervisorPlan(
        allowed=True,
        reason="Travel request",
        selected_agents=["flight_agent", "itinerary_agent"],
        trip_constraints=TripConstraints(destination="Rishikesh"),
        destination_iata="",
        reasoning="Destination has no airport code",
    )
    monkeypatch.setattr(backend, "call_llm", lambda *args, **kwargs: plan)
    result = supervisor_agent(
        _state(user_query="Plan a 3 day trip to Rishikesh from Delhi under 4000 INR")
    )
    assert not backend.has_airport(result["trip_constraints"]["destination_iata"])
    assert "flight_agent" not in result["selected_agents"]
    rendered = backend.render_flight_section(
        {"data_type": "unavailable", "error": "service could not be reached"},
        result["trip_constraints"],
        requested="flight_agent" in result["selected_agents"],
    )
    assert "## Getting there" in rendered
    assert "Dehradun (DED) for Rishikesh" in rendered
    assert "service could not be reached" not in rendered


def test_city_lookup_finds_varanasi_airport_without_supervisor_iata(monkeypatch):
    plan = SupervisorPlan(
        allowed=True,
        reason="Travel request",
        selected_agents=["itinerary_agent"],
        trip_constraints=TripConstraints(destination="Varanasi"),
        destination_iata="",
        reasoning="Trip request",
    )
    monkeypatch.setattr(backend, "call_llm", lambda *args, **kwargs: plan)

    result = supervisor_agent(
        _state(user_query="Plan a 3 day trip to Varanasi")
    )

    assert result["trip_constraints"]["destination_iata"] == "VNS"
    assert backend.has_airport(result["trip_constraints"]["destination_iata"])
    rendered = backend.render_flight_section(
        {"data_type": "unavailable", "error": "flight lookup failed"},
        result["trip_constraints"],
        requested=False,
    )
    assert "## Flights" in rendered
    assert "Varanasi" in rendered
    assert "VNS" in rendered
    assert "no commercial airport" not in rendered
    assert "flight lookup failed" not in rendered


@pytest.mark.parametrize(
    ("destination", "expected_code"),
    [
        ("Delhi", "DEL"),
        ("New Delhi", "DEL"),
        ("Bengaluru", "BLR"),
        ("Bangalore", "BLR"),
        ("Mumbai", "BOM"),
        ("Bombay", "BOM"),
        ("Varanasi, Uttar Pradesh, India", "VNS"),
        ("Varanassi", "VNS"),
    ],
)
def test_destination_airport_lookup_handles_aliases_and_variants(
    destination,
    expected_code,
):
    assert backend._resolve_destination_airport(destination) == expected_code


def test_destination_airport_lookup_rejects_mismatched_supervisor_code():
    assert backend._resolve_destination_airport("Varanasi", "DEL") == "VNS"


def test_destination_region_lookup_finds_goa_commercial_airport():
    code = backend._resolve_destination_airport("Goa")
    assert code in {"GOI", "GOX"}
    rendered = backend.render_flight_section(
        {"data_type": "unavailable", "error": "flight data unavailable"},
        {"destination": "Goa", "destination_iata": code},
        requested=True,
    )
    assert "## Flights" in rendered
    assert code in rendered
    assert "no commercial airport" not in rendered.casefold()
    assert "Airport availability" not in rendered


def test_unknown_destination_does_not_make_unverified_no_airport_claim():
    rendered = backend.render_flight_section(
        {"data_type": "unavailable", "error": "flight service failed"},
        {"destination": "Remote Example Place", "destination_iata": ""},
    )
    assert "Airport availability for Remote Example Place could not be verified" in rendered
    assert "has no commercial airport" not in rendered
    assert "flight service failed" not in rendered


def test_country_destination_explains_multiple_airports():
    rendered = backend.render_flight_section(
        {"data_type": "unavailable", "error": "flight service failed"},
        {"destination": "Japan", "destination_iata": ""},
    )
    assert "Japan has multiple commercial airports" in rendered
    assert "Choose an arrival city" in rendered
    assert "could not be verified from the airport directory" not in rendered
    assert "service could not be reached" not in rendered


def test_budget_prompt_uses_ground_transport_for_low_inr_budget(monkeypatch):
    observed_prompts = []
    budget = TripBudget(
        transport=300,
        stay=900,
        food=500,
        activities=200,
        local_travel=100,
        buffer=100,
        currency="INR",
        assumptions=[],
    )

    def fake_call(messages, **kwargs):
        observed_prompts.append(backend._message_text(messages))
        return budget

    monkeypatch.setattr(backend, "call_llm", fake_call)
    backend.budget_agent(
        _state(
            user_query="Plan a trip to Delhi under 4000 INR",
            trip_constraints={
                "destination": "Delhi",
                "destination_iata": "DEL",
                "budget": "under 4000 INR",
            },
        )
    )
    assert "bus/train (do not estimate a flight)" in observed_prompts[0]


def test_flight_status_reports_not_requested():
    assert backend.render_flight_status({}, requested=False) == "Flights not requested"


def test_budget_section_is_omitted_if_budget_agent_did_not_run():
    state = _state(selected_agents=["itinerary_agent"])
    itinerary = backend._assemble_itinerary(state, "## Day-by-day plan\n### Day 1 - Arrival")
    assert "## Budget" not in itinerary
    assert "No user budget was provided" not in itinerary


def test_set_budget_never_renders_as_missing():
    section = render_budget_section(
        {
            "transport": 1000,
            "stay": 1000,
            "food": 500,
            "activities": 500,
            "local_travel": 300,
            "buffer": 200,
            "currency": "INR",
            "budget_total": 3500,
            "budget_gap": 500,
        },
        "under 4000 INR",
    )
    assert "No user budget was provided" not in section
    assert "Within the stated budget" in section


def test_budget_validation_rejects_zero_categories():
    budget = TripBudget(
        transport=100,
        stay=0,
        food=50,
        activities=25,
        local_travel=15,
        buffer=10,
        currency="INR",
        assumptions=[],
    )
    assert "stay must be greater than zero." in backend._budget_validation_problems(
        budget,
        None,
    )


def test_budget_agent_regenerates_zero_category_once(monkeypatch):
    valid_budget = TripBudget(
        transport=100,
        stay=200,
        food=50,
        activities=25,
        local_travel=15,
        buffer=10,
        currency="INR",
        assumptions=[],
    )
    invalid_budget = valid_budget.model_copy(update={"stay": 0})
    responses = iter([invalid_budget, valid_budget])
    monkeypatch.setattr(
        backend,
        "call_llm",
        lambda *args, **kwargs: next(responses),
    )
    result = budget_agent(
        _state(trip_constraints={"destination": "Rome", "budget": "₹1,000"})
    )
    assert result["budget_total"] == 400.0
    assert result["llm_calls"] == 2


def test_flight_agent_renders_airport_and_availability_without_llm(monkeypatch):
    monkeypatch.setattr(
        backend,
        "_search_live_flights",
        lambda origin, destination: "No live flight data found",
    )
    monkeypatch.setattr(
        backend,
        "call_llm",
        lambda *args, **kwargs: pytest.fail("Flight data must be rendered in Python."),
    )
    result = backend.flight_agent(
        _state(
            selected_agents=["flight_agent"],
            trip_constraints={
                "destination": "Rome",
                "origin": "Delhi",
                "origin_iata": "DEL",
                "destination_iata": "FCO",
            }
        )
    )
    assert "New Delhi (DEL)" in result["flight_results"]
    assert "Rome (FCO)" in result["flight_results"]
    assert "Airline and flight names are unavailable" in result["flight_results"]


def test_supervisor_defaults_boarding_to_delhi(monkeypatch):
    plan = SupervisorPlan(
        allowed=True,
        reason="Travel request",
        selected_agents=["itinerary_agent"],
        trip_constraints=TripConstraints(destination="Rome"),
        origin_iata="",
        destination_iata="FCO",
        reasoning="No boarding location provided",
    )
    monkeypatch.setattr(backend, "call_llm", lambda *args, **kwargs: plan)

    result = supervisor_agent(_state(user_query="Plan a trip to Rome"))

    assert result["trip_constraints"]["origin"] == "Delhi"
    assert result["trip_constraints"]["origin_iata"] == "DEL"
    assert "flight_agent" in result["selected_agents"]


def test_flight_agent_displays_names_only_from_live_route_results(monkeypatch):
    monkeypatch.setattr(
        backend,
        "_search_live_flights",
        lambda origin, destination: (
            "Live flights from DEL to FCO\n\n"
            "Airline: Example Air\nFlight: EX123\nStatus: Active"
        ),
    )
    result = backend.flight_agent(
        _state(
            selected_agents=["flight_agent"],
            trip_constraints={
                "destination": "Rome",
                "origin": "Delhi",
                "origin_iata": "DEL",
                "destination_iata": "FCO",
            },
        )
    )
    assert result["flight_data"]["data_type"] == "live"
    assert "Example Air - EX123" in result["flight_results"]
    assert "Delhi" in result["flight_results"]
    assert "Ticket fares and future schedules are unavailable" not in result["flight_results"]


def test_assembled_itinerary_displays_flight_and_hotel_sections():
    state = _state(
        selected_agents=["itinerary_agent", "hotel_agent", "flight_agent"],
        flight_data={
            "data_type": "unavailable",
            "error": "flight service could not be reached",
        },
        trip_constraints={
            "destination": "Rome",
            "duration_days": 2,
            "origin_iata": "DEL",
            "destination_iata": "FCO",
        },
        hotel_data={
            "data_type": "live",
            "payload": {
                "results": [
                    {"title": "Central stay", "url": "https://booking.com/hotel"}
                ]
            },
        },
    )
    itinerary = backend._assemble_itinerary(state, "## Day-by-day plan\n### Day 1 - Arrival")
    assert itinerary.count("## Trip itinerary") == 1
    assert "## Flights" in itinerary
    assert "Indira Gandhi International Airport" in itinerary
    assert "## Stay options" in itinerary
    assert "Central stay" in itinerary
    assert "authoritative" not in itinerary.lower()
    assert "do not alter" not in itinerary.lower()


def test_rendered_budget_formats_inr_and_stay_options_exclude_social_sites():
    section = render_budget_section(
        {
            "transport": 5000,
            "stay": 5000,
            "food": 5000,
            "activities": 5000,
            "local_travel": 5000,
            "buffer": 5000,
            "currency": "INR",
            "budget_total": 30000,
        },
        "₹60,000",
    )
    assert "₹5,000" in section
    assert "₹30,000" in section
    stay = backend.render_stay_section(
        {
            "data_type": "live",
            "payload": {
                "results": [
                    {"title": "Social listing", "url": "https://instagram.com/a"},
                    {"title": "Central stay", "url": "https://booking.com/hotel"},
                ]
            },
        }
    )
    assert "Central stay" in stay
    assert "instagram.com" not in stay
    assert "Prices not verified" in stay


def test_stay_filter_keeps_booking_rupee_price_and_drops_spanish_usd():
    hotels = backend._condensed_hotels(
        {
            "payload": {
                "results": [
                    {
                        "title": "Hoteles baratos desde $99 por noche",
                        "url": "https://www.skyscanner.net/hoteles/",
                        "content": "Alojamiento desde USD 99",
                    },
                    {
                        "title": "Riverside Stay",
                        "url": "https://www.booking.com/hotel/in/riverside.html",
                        "content": "Rates from ₹1,260 per night",
                    },
                ]
            }
        }
    )
    assert len(hotels) == 1
    assert hotels[0]["title"] == "Riverside Stay"
    assert hotels[0]["price_value"] == 1260
    assert hotels[0]["price_snippet"] == (
        "from ₹1,260 per night (source: booking.com)"
    )


def test_call_llm_trims_hotel_data_before_sending():
    class MockModel:
        def invoke(self, messages):
            self.messages = messages
            return AIMessage(content="ok")

    model = MockModel()
    messages = [
        HumanMessage(
            content=(
                "Plan a trip\n"
                '<tool_data source="hotel_search">'
                + ("hotel result " * 9000)
                + "</tool_data>"
            )
        )
    ]
    backend.call_llm(messages, model=model)
    prompt = model.messages[0].content
    assert backend.estimate_tokens(prompt) <= 5500
    assert "[tool data trimmed" in prompt


def test_call_llm_never_sends_prompt_over_6000_tokens():
    class MockModel:
        called = False

        def invoke(self, messages):
            self.called = True

    model = MockModel()
    with pytest.raises(ValueError, match="no request was sent"):
        backend.call_llm("x" * 24004, model=model)
    assert model.called is False


def test_call_llm_honors_bounded_retry_after_for_429(monkeypatch):
    class MockResponse:
        status_code = 429
        headers = {"Retry-After": "45"}

    class RateLimited(Exception):
        response = MockResponse()

    class MockModel:
        attempts = 0

        def invoke(self, messages):
            self.attempts += 1
            if self.attempts == 1:
                raise RateLimited("limited")
            return AIMessage(content="ok")

    sleeps = []
    monkeypatch.setattr(backend.time, "sleep", sleeps.append)
    model = MockModel()
    backend.call_llm("short prompt", model=model, retries=2)
    assert model.attempts == 2
    assert sleeps == [30.0]


def test_call_llm_does_not_retry_413(monkeypatch):
    class TooLarge(Exception):
        status_code = 413

    class MockModel:
        attempts = 0

        def invoke(self, messages):
            self.attempts += 1
            raise TooLarge("request too large")

    monkeypatch.setattr(backend.time, "sleep", lambda _: pytest.fail("No retry"))
    model = MockModel()
    with pytest.raises(TooLarge):
        backend.call_llm("short prompt", model=model, retries=2)
    assert model.attempts == 1


def test_token_window_waits_until_old_usage_expires(monkeypatch):
    current_time = [100.0]
    sleeps = []
    monkeypatch.setattr(backend.time, "monotonic", lambda: current_time[0])

    def sleep(seconds):
        sleeps.append(seconds)
        current_time[0] += seconds

    monkeypatch.setattr(backend.time, "sleep", sleep)
    backend._token_history.append((41.0, 6800))
    backend._reserve_tokens(500)
    assert sleeps == [1.0]
    assert backend._token_history == deque([(101.0, 500)])


def test_timed_node_records_prompt_token_estimate(monkeypatch):
    plan = SupervisorPlan(
        allowed=True,
        reason="Travel request",
        selected_agents=["itinerary_agent"],
        trip_constraints=TripConstraints(destination="Rome"),
        reasoning="Valid trip",
    )
    def fake_llm(messages, **kwargs):
        backend._node_prompt_tokens.set(
            backend._node_prompt_tokens.get()
            + backend.estimate_tokens(backend._message_text(messages))
        )
        return plan

    monkeypatch.setattr(backend, "call_llm", fake_llm)
    result = supervisor_agent(_state())
    assert result["node_timings"]["supervisor_agent"]["estimated_prompt_tokens"] > 0


def test_node_timings_include_finish_reason_and_completion_tokens():
    class MockModel:
        def bind(self, **kwargs):
            return self

        def invoke(self, messages):
            return AIMessage(
                content="done",
                response_metadata={"finish_reason": "stop"},
                usage_metadata={
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "total_tokens": 5,
                },
            )

    @backend._timed_node
    def measured_node(state):
        backend.call_llm("small prompt", model=MockModel(), max_tokens=50)
        return {}

    result = measured_node(_state())
    metric = result["node_timings"]["measured_node"]["llm_calls"][0]
    assert metric["finish_reason"] == "stop"
    assert metric["completion_tokens"] == 3


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"approval_status": "approved", "revision_count": 0}, "final_agent"),
        ({"approval_status": "revision_requested", "revision_count": 1}, "revise_itinerary"),
        ({"approval_status": "revision_requested", "revision_count": 3}, "revision_limit_reached"),
    ],
)
def test_route_after_review(state, expected):
    assert route_after_review(
        {"max_revisions": 3, **state}
    ) == expected


def test_validate_itinerary_catches_wrong_day_count():
    problems = validate_itinerary(
        _state(
            day_by_day_plan=(
                "Day-by-day plan\nDay 1: Visit the center\n\n"
            )
        )
    )
    assert any("Expected day numbers 1 through 2" in problem for problem in problems)


@pytest.mark.parametrize(
    "day_section",
    [
        "## Day\u2011by\u2011day plan\n### Day 1 - A\n- Walk\n### Day 2 - B\n- Rest",
        "## Day-by-day plan\n| Day 1 | activity |\n| Day 2 | activity |",
        "## Day-by-day plan\n### Day 1 - A\n### Day 2 - B",
    ],
)
def test_validator_accepts_normalized_headings_and_rows(day_section):
    assert validate_itinerary(
        _state(day_by_day_plan=day_section)
    ) == []


def test_parse_amount_handles_indian_rupee_grouping():
    assert parse_amount("₹30 000") == 30000.0
    assert parse_amount("30,000") == 30000.0
    assert parse_amount("30\u00a0000") == 30000.0


def test_rendered_budget_total_matches_budget_total():
    section = render_budget_section(
        {
            **_state()["budget_breakdown"],
            "budget_total": 10.0,
            "budget_gap": 5.0,
        },
        "EUR 15 000",
    )
    assert "| **Total** | **10 EUR** |" in section
    assert "Within the stated budget by 5 EUR" in section


def test_failed_budget_estimate_does_not_render_empty_markdown_table():
    section = backend._assemble_itinerary(
        _state(
            selected_agents=["budget_agent", "itinerary_agent"],
            budget_breakdown={},
            budget_total=None,
            budget_error="The cost estimate could not be generated. Please retry.",
            day_by_day_plan="## Day-by-day plan\n### Day 1 - Arrival",
        ),
        "## Day-by-day plan\n### Day 1 - Arrival",
    )
    assert "## Budget\nBudget estimate unavailable." in section
    assert "The cost estimate could not be generated. Please retry." in section
    assert "| Category |" not in section
    assert "Not available" not in section


def test_render_budget_asserts_computed_total_matches_categories():
    with pytest.raises(AssertionError, match="budget_total"):
        render_budget_section(
            {
                **_state()["budget_breakdown"],
                "budget_total": 11.0,
            },
            None,
        )


def test_weather_and_flight_renderers_use_source_facts():
    weather = render_weather_section(
        {
            "data_type": "live",
            "payload": {
                "forecast": [
                    {
                        "date": "2026-10-29",
                        "min_temp_c": 13.25,
                        "max_temp_c": 21.5,
                        "dominant_condition": "light rain",
                    }
                ]
            },
        }
    )
    assert "2026-10-29: 13.25 to 21.5 C, light rain" in weather
    assert "Flight data: unavailable" in render_flight_status(
        {"data_type": "unavailable"}
    )


def test_itinerary_prompt_is_compact():
    state = _state(
        hotel_data={
            "data_type": "live",
            "payload": {
                "results": [
                    {
                        "title": f"Hotel {index}",
                        "url": f"https://example{index}.com",
                        "content": "Details " * 200,
                    }
                    for index in range(5)
                ]
            },
        },
        weather_data={
            "data_type": "live",
            "payload": {
                "forecast": [
                    {
                        "date": f"2026-10-{day:02d}",
                        "min_temp_c": 10,
                        "max_temp_c": 20,
                        "dominant_condition": "sunny",
                    }
                    for day in range(1, 6)
                ]
            },
        },
    )
    assert backend.estimate_tokens(backend._itinerary_prompt(state)) < 2500


def test_assembled_itinerary_strips_prompt_artifacts():
    output = backend._assemble_itinerary(
        _state(
            flight_data={"data_type": "unavailable"},
            day_by_day_plan="## Day-by-day plan\n### Day 1 - Walk",
        ),
        "## Day-by-day plan\n### Day 1 - Walk [source=fixture] authoritative",
    )
    assert "[source=" not in output
    assert "authoritative" not in output.lower()


def test_length_finish_retries_once_then_itinerary_fails(monkeypatch):
    class MockModel:
        def __init__(self):
            self.max_token_settings = []
            self.attempts = 0

        def bind(self, **kwargs):
            self.max_token_settings.append(kwargs["max_tokens"])
            return self

        def invoke(self, messages):
            self.attempts += 1
            return AIMessage(
                content="truncated",
                response_metadata={"finish_reason": "length"},
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 1800,
                    "total_tokens": 1810,
                },
            )

    model = MockModel()
    with pytest.raises(backend.LLMTruncatedError):
        backend.call_llm("short prompt", model=model, max_tokens=1800)
    assert model.attempts == 2
    assert model.max_token_settings == [1800, 2300]

    monkeypatch.setattr(
        backend,
        "call_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            backend.LLMTruncatedError("Still truncated")
        ),
    )
    result = itinerary_agent(_state())
    assert result["itinerary_failed"] is True
    assert route_after_itinerary(result) == "draft_failed"


def test_reject_revise_approve_finalize_graph(monkeypatch):
    approved_draft = (
        "## Day-by-day plan\n"
        "### Day 1 - Explore Rome\n- Visit the center\n"
        "### Day 2 - Museums\n- Visit a museum district"
    )

    def fake_llm(messages, retries=2, model=None, max_tokens=None):
        if "Revise the travel itinerary" in messages[0].content:
            return AIMessage(content=approved_draft)
        if "concise, grounded travel recommendations" in messages[0].content:
            return AIMessage(content="Check local conditions before departure.")
        return AIMessage(content=approved_draft)

    monkeypatch.setattr(backend, "call_llm", fake_llm)

    def initial_draft(state):
        return {
            "itinerary": approved_draft,
            "day_by_day_plan": approved_draft,
        }

    workflow = StateGraph(TravelState)
    workflow.add_node("draft", initial_draft)
    workflow.add_node("human_approval", human_approval_agent)
    workflow.add_node("revise_itinerary", revise_itinerary)
    workflow.add_node("final_agent", final_agent)
    workflow.add_edge(START, "draft")
    workflow.add_edge("draft", "human_approval")
    workflow.add_conditional_edges(
        "human_approval",
        route_after_review,
        {
            "final_agent": "final_agent",
            "revise_itinerary": "revise_itinerary",
            "revision_limit_reached": END,
        },
    )
    workflow.add_edge("revise_itinerary", "human_approval")
    workflow.add_edge("final_agent", END)
    app = workflow.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "review-loop-test"}}

    state = _state(
        itinerary="",
        day_by_day_plan="",
        budget_total=10.0,
        max_revisions=3,
        approval_status="pending",
    )
    first = app.invoke(state, config=config)
    assert first["__interrupt__"]

    second = app.invoke(
        Command(resume={"approved": False, "feedback": "Add more detail"}),
        config=config,
    )
    assert second["__interrupt__"]
    assert second["revision_count"] == 1

    third = app.invoke(
        Command(resume={"approved": True, "feedback": ""}),
        config=config,
    )
    assert third["approval_status"] == "approved"
    assert "Check local conditions" in third["final_response"]
    assert "2026-10" not in third["final_response"]


def test_approval_endpoint_rejects_thread_not_waiting_for_review(
    monkeypatch,
    authenticated_client,
):
    monkeypatch.setattr(
        api,
        "get_trip_by_owned_thread",
        lambda db, user_id, thread_id: SimpleNamespace(
            id=uuid.uuid4(),
            thread_id=thread_id,
        ),
    )
    monkeypatch.setattr(
        api.travel_graph,
        "get_state",
        lambda config: type("Snapshot", (), {"next": ("final_agent",)})(),
    )
    response = authenticated_client.post(
        "/api/travel/approve",
        json={
            "thread_id": "already-finished",
            "approved": True,
            "feedback": "",
        },
    )
    assert response.status_code == 409
    assert response.json()["success"] is False


def test_approval_endpoint_preserves_selected_voice_language(
    monkeypatch,
    authenticated_client,
):
    monkeypatch.setattr(
        api,
        "get_trip_by_owned_thread",
        lambda db, user_id, thread_id: SimpleNamespace(
            id=uuid.uuid4(),
            thread_id=thread_id,
        ),
    )
    monkeypatch.setattr(api, "record_workflow_update", lambda *args, **kwargs: None)
    received = {}
    monkeypatch.setattr(
        api.travel_graph,
        "get_state",
        lambda config: type("Snapshot", (), {"next": ("human_approval",)})(),
    )

    def fake_resume(**kwargs):
        received.update(kwargs)
        return {"thread_id": kwargs["thread_id"], "answer": "यात्रा तैयार है"}

    monkeypatch.setattr(api, "resume_travel_agent", fake_resume)
    response = authenticated_client.post(
        "/api/travel/approve",
        json={
            "thread_id": "voice-session",
            "approved": True,
            "language": "hi",
        },
    )
    assert response.status_code == 200
    assert received["preferred_language"] == "hi"
    assert received["thread_id"] == "voice-session"


def test_travel_endpoint_ignores_client_thread_id(
    monkeypatch,
    authenticated_client,
):
    received = {}

    def fake_run(**kwargs):
        received.update(kwargs)
        return {"thread_id": "server-generated"}

    monkeypatch.setattr(api, "run_travel_agent", fake_run)
    response = authenticated_client.post(
        "/api/travel",
        json={
            "message": "Plan a trip to Rome",
            "thread_id": "client-chosen",
            "language": "hi",
        },
    )
    assert response.status_code == 200
    assert "thread_id" not in received
    assert received["preferred_language"] == "hi"


def test_itinerary_prompt_and_rendering_follow_selected_language():
    state = {
        "preferred_language": "hi",
        "trip_constraints": {"destination": "Jaipur", "duration_days": 1},
        "selected_agents": [],
        "weather_requested": False,
    }
    prompt = backend._itinerary_prompt(state)
    rendered = backend._assemble_itinerary(
        state,
        "## Day-by-day plan\n### Day 1 - Arrival\n- Explore the old city.",
    )
    assert "natural Hindi" in prompt
    assert "## यात्रा कार्यक्रम" in rendered
    assert "## मौसम" in rendered


def test_voice_transcription_rejects_unsupported_audio_format(authenticated_client):
    response = authenticated_client.post(
        "/api/voice/transcribe",
        files={"audio": ("voice.txt", b"not audio", "text/plain")},
        data={"language": "en"},
    )
    assert response.status_code == 415


def test_voice_transcription_uses_groq_without_persisting_audio(
    monkeypatch,
    authenticated_client,
):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"text": "Plan a trip to Jaipur", "language": "en"}

    sent = {}

    def fake_post(url, **kwargs):
        sent["url"] = url
        sent["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setenv("GROQ_API_KEY", "test-token")
    monkeypatch.setattr(api.requests, "post", fake_post)
    response = authenticated_client.post(
        "/api/voice/transcribe",
        files={"audio": ("voice.webm", b"\x1a\x45\xdf\xa3audio", "audio/webm")},
        data={"language": "auto"},
    )

    assert response.status_code == 200
    assert response.json()["text"] == "Plan a trip to Jaipur"
    assert response.json()["detected_language"] == "en"
    assert sent["url"].endswith("/audio/transcriptions")
    assert sent["kwargs"]["data"] == {
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",
    }
    assert sent["kwargs"]["files"]["file"][0] == "voice-recording.webm"
    assert "test-token" not in response.text


def test_voice_transcription_requires_provider_configuration(
    monkeypatch,
    authenticated_client,
):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = authenticated_client.post(
        "/api/voice/transcribe",
        files={"audio": ("voice.webm", b"\x1a\x45\xdf\xa3audio", "audio/webm")},
        data={"language": "hi"},
    )
    assert response.status_code == 503
    assert "Text input is still available" in response.json()["detail"]


def test_fastapi_lifespan_closes_connection_pool(monkeypatch):
    closed = []
    monkeypatch.setattr(api.connection_pool, "close", lambda: closed.append(True))
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
    assert closed == [True]
