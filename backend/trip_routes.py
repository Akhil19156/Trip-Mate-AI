import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .auth import require_csrf, get_current_user
from .database import get_db
from .graph import travel_graph
from .models import Trip, User
from .schemas_auth import (
    ConversationResponse,
    TripCreate,
    TripResponse,
    TripUpdate,
)
from .trip_service import get_owned_trip

router = APIRouter(prefix="/api/trips", tags=["travel history"])


@router.post("", status_code=201, response_model=TripResponse)
def create_trip(
    payload: TripCreate,
    user: User = Depends(require_csrf),
    db: Session = Depends(get_db),
):
    if (
        payload.travel_start_date
        and payload.travel_end_date
        and payload.travel_end_date < payload.travel_start_date
    ):
        raise HTTPException(status_code=422, detail="End date must be on or after start date.")
    trip = Trip(user_id=user.id, **payload.model_dump())
    db.add(trip)
    db.commit()
    db.refresh(trip)
    return trip


@router.get("", response_model=list[TripResponse])
def list_trips(
    search: str | None = Query(default=None, max_length=160),
    status: str | None = Query(default=None, max_length=40),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    statement = select(Trip).where(Trip.user_id == user.id)
    if search:
        statement = statement.where(
            Trip.destination.ilike(f"%{search.strip()}%")
            | Trip.title.ilike(f"%{search.strip()}%")
        )
    if status:
        statement = statement.where(Trip.status == status)
    return list(
        db.scalars(
            statement.order_by(Trip.created_at.desc()).offset(offset).limit(limit)
        ).all()
    )


@router.get("/summary")
def summarize_trips(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    total = db.scalar(
        select(func.count()).select_from(Trip).where(Trip.user_id == user.id)
    )
    recent = db.scalars(
        select(Trip)
        .where(Trip.user_id == user.id)
        .order_by(Trip.created_at.desc())
        .limit(5)
    ).all()
    return {
        "total_trips": total or 0,
        "recent_trips": [TripResponse.model_validate(trip) for trip in recent],
    }


@router.get("/{trip_id}")
def get_trip(
    trip_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    trip = get_owned_trip(db, user.id, trip_id, with_conversations=True)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found.")
    return {
        "trip": TripResponse.model_validate(trip),
        "conversations": [
            ConversationResponse.model_validate(item)
            for item in sorted(trip.conversations, key=lambda message: message.created_at)
        ],
    }


@router.put("/{trip_id}", response_model=TripResponse)
def update_trip(
    trip_id: uuid.UUID,
    payload: TripUpdate,
    user: User = Depends(require_csrf),
    db: Session = Depends(get_db),
):
    trip = get_owned_trip(db, user.id, trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found.")
    values = payload.model_dump(exclude_unset=True)
    start_date = values.get("travel_start_date", trip.travel_start_date)
    end_date = values.get("travel_end_date", trip.travel_end_date)
    if start_date and end_date and end_date < start_date:
        raise HTTPException(status_code=422, detail="End date must be on or after start date.")
    for field, value in values.items():
        setattr(trip, field, value)
    db.commit()
    db.refresh(trip)
    return trip


@router.delete("/{trip_id}", status_code=204)
def delete_trip(
    trip_id: uuid.UUID,
    user: User = Depends(require_csrf),
    db: Session = Depends(get_db),
):
    trip = get_owned_trip(db, user.id, trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found.")
    db.delete(trip)
    db.commit()
    return None


@router.post("/{trip_id}/resume")
def resume_trip(
    trip_id: uuid.UUID,
    user: User = Depends(require_csrf),
    db: Session = Depends(get_db),
):
    trip = get_owned_trip(db, user.id, trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found.")
    if not trip.thread_id:
        raise HTTPException(status_code=409, detail="This saved trip has no resumable planning session.")
    config = {"configurable": {"thread_id": trip.thread_id}}
    snapshot = travel_graph.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="The saved planning session is unavailable.")
    return {
        "trip": TripResponse.model_validate(trip),
        "requires_approval": "human_approval" in snapshot.next,
        "next": list(snapshot.next),
    }
