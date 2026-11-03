from datetime import date, datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

TripStatus = Literal[
    "draft",
    "pending_approval",
    "revision_requested",
    "revision_failed",
    "revision_limit_reached",
    "approved",
    "completed",
    "saved",
]


class SignupRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)

    @model_validator(mode="after")
    def validate_password(self):
        password = self.password
        if not any(char.islower() for char in password):
            raise ValueError("Password must contain a lowercase letter.")
        if not any(char.isupper() for char in password):
            raise ValueError("Password must contain an uppercase letter.")
        if not any(char.isdigit() for char in password):
            raise ValueError("Password must contain a number.")
        if not any(not char.isalnum() for char in password):
            raise ValueError("Password must contain a symbol.")
        return self


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    id: UUID
    name: str
    email: EmailStr
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PreferencesUpdate(BaseModel):
    preferred_language: Literal["en", "hi", "hinglish"] = "en"
    preferred_currency: str = Field(default="INR", min_length=3, max_length=8)
    travel_style: str = Field(default="", max_length=40)
    accommodation_preferences: str = Field(default="", max_length=2000)
    preferred_budget: Decimal | None = Field(default=None, ge=0, le=100_000_000)

    @field_validator("preferred_currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized.isalpha():
            raise ValueError("Currency must be an ISO-style alphabetic code.")
        return normalized


class ProfileUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class PreferencesResponse(PreferencesUpdate):
    id: UUID | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class TripCreate(BaseModel):
    title: str = Field(default="", max_length=200)
    origin: str = Field(default="", max_length=160)
    destination: str = Field(min_length=1, max_length=160)
    travel_start_date: date | None = None
    travel_end_date: date | None = None
    budget: Decimal | None = Field(default=None, ge=0, le=100_000_000)
    status: Literal["saved", "draft"] = "saved"
    itinerary: str = Field(default="", max_length=100_000)


class TripUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    origin: str | None = Field(default=None, max_length=160)
    destination: str | None = Field(default=None, min_length=1, max_length=160)
    travel_start_date: date | None = None
    travel_end_date: date | None = None
    budget: Decimal | None = Field(default=None, ge=0, le=100_000_000)
    itinerary: str | None = Field(default=None, max_length=100_000)


class TripResponse(BaseModel):
    id: UUID
    user_id: UUID
    thread_id: str | None
    title: str
    origin: str
    destination: str
    travel_start_date: date | None
    travel_end_date: date | None
    budget: Decimal | None
    status: str
    itinerary: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationResponse(BaseModel):
    id: UUID
    user_id: UUID
    role: str
    message: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
