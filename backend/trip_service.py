import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .models import Trip, TripConversation, User, UserPreference


def preference_context(user: User) -> dict[str, object]:
    preferences = user.preferences
    if preferences is None:
        return {
            "preferred_language": "en",
            "preferred_currency": "INR",
            "travel_style": "",
            "preferred_budget": None,
            "accommodation_preferences": "",
        }
    return {
        "preferred_language": preferences.preferred_language,
        "preferred_currency": preferences.preferred_currency,
        "travel_style": preferences.travel_style,
        "preferred_budget": (
            float(preferences.preferred_budget)
            if preferences.preferred_budget is not None
            else None
        ),
        "accommodation_preferences": preferences.accommodation_preferences,
    }


def get_owned_trip(
    db: Session,
    user_id: uuid.UUID,
    trip_id: uuid.UUID,
    *,
    with_conversations: bool = False,
) -> Trip | None:
    statement = select(Trip).where(
        Trip.id == trip_id,
        Trip.user_id == user_id,
    )
    if with_conversations:
        statement = statement.options(selectinload(Trip.conversations))
    return db.scalar(statement)


def get_trip_by_owned_thread(
    db: Session,
    user_id: uuid.UUID,
    thread_id: str,
) -> Trip | None:
    return db.scalar(
        select(Trip).where(
            Trip.user_id == user_id,
            Trip.thread_id == thread_id,
        )
    )


def _budget_from_constraints(constraints: dict[str, object]) -> Decimal | None:
    import re

    raw = str(constraints.get("budget") or "")
    digits = re.sub(r"[^\d.]", "", raw)
    if not digits:
        return None
    try:
        return Decimal(digits)
    except InvalidOperation:
        return None


def save_workflow_trip(
    db: Session,
    user: User,
    user_message: str,
    result: dict[str, object],
) -> Trip | None:
    if not result.get("guardrail_allowed"):
        return None
    thread_id = str(result.get("thread_id") or "")
    constraints = result.get("trip_constraints")
    if not thread_id or not isinstance(constraints, dict):
        return None
    destination = str(constraints.get("destination") or "").strip()
    if not destination:
        return None
    trip = Trip(
        user_id=user.id,
        thread_id=thread_id,
        title=f"Trip to {destination}",
        origin=str(constraints.get("origin") or ""),
        destination=destination,
        budget=_budget_from_constraints(constraints),
        status=(
            "pending_approval"
            if result.get("requires_approval")
            else str(result.get("approval_status") or "completed")
        ),
        itinerary=str(result.get("itinerary") or result.get("answer") or ""),
    )
    db.add(trip)
    db.flush()
    db.add_all(
        [
            TripConversation(
                trip_id=trip.id,
                user_id=user.id,
                role="user",
                message=user_message,
            ),
            TripConversation(
                trip_id=trip.id,
                user_id=user.id,
                role="assistant",
                message=trip.itinerary or str(result.get("answer") or ""),
            ),
        ]
    )
    return trip


def record_workflow_update(
    db: Session,
    trip: Trip,
    user: User,
    result: dict[str, object],
    feedback: str = "",
) -> None:
    trip.itinerary = str(result.get("itinerary") or result.get("answer") or trip.itinerary)
    if result.get("approval_status"):
        trip.status = str(result["approval_status"])
    elif result.get("requires_approval"):
        trip.status = "pending_approval"
    else:
        trip.status = "completed"
    trip.updated_at = datetime.now(timezone.utc)
    message = (
        feedback.strip()
        if feedback.strip()
        else str(result.get("answer") or "")
    )
    if message:
        db.add(
            TripConversation(
                trip_id=trip.id,
                user_id=user.id,
                role="user" if feedback.strip() else "assistant",
                message=message,
            )
        )
    assistant_message = str(result.get("answer") or "")
    if feedback.strip() and assistant_message:
        db.add(
            TripConversation(
                trip_id=trip.id,
                user_id=user.id,
                role="assistant",
                message=assistant_message,
            )
        )


def ensure_user_preferences(db: Session, user: User) -> UserPreference:
    if user.preferences is None:
        user.preferences = UserPreference(user_id=user.id)
        db.add(user.preferences)
        db.flush()
    return user.preferences
