from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool


AgentName = Literal[
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


class TripConstraints(BaseModel):
    destination: str = ""
    origin: str = ""
    origin_iata: str = ""
    destination_iata: str = ""
    duration_days: int | None = None
    budget: str = ""
    travel_style: str = ""
    special_preferences: list[str] = Field(default_factory=list)


class SupervisorPlan(BaseModel):
    model_config = ConfigDict(strict=True)

    allowed: StrictBool
    reason: str
    selected_agents: list[AgentName]
    trip_constraints: TripConstraints
    origin_iata: str = ""
    destination_iata: str = ""
    reasoning: str


class ToolResult(BaseModel):
    source: str
    retrieved_at: str
    data_type: Literal["live", "estimate", "unavailable"]
    payload: Any
    error: str | None = None


class TripBudget(BaseModel):
    transport: float
    stay: float
    food: float
    activities: float
    local_travel: float
    buffer: float
    currency: str
    assumptions: list[str]

    @property
    def total(self) -> float:
        return (
            self.transport
            + self.stay
            + self.food
            + self.activities
            + self.local_travel
            + self.buffer
        )
