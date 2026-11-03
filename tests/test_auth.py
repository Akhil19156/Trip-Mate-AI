import uuid

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.app as api
from backend.auth import ACCESS_COOKIE, REFRESH_COOKIE, get_current_user, require_csrf
from backend.database import get_db
from backend.models import Base, Trip, User

PASSWORD = "StrongPassword123!"
JWT_SECRET = "test-only-secret-key-at-least-32-bytes"


@pytest.fixture
def auth_client(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", JWT_SECRET)
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    test_session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db():
        with test_session() as session:
            yield session

    monkeypatch.setitem(api.app.dependency_overrides, get_db, override_get_db)
    with TestClient(api.app) as client:
        yield client, test_session
    engine.dispose()


def signup(client: TestClient, email: str):
    return client.post(
        "/api/auth/signup",
        json={"name": "Trip Mate User", "email": email, "password": PASSWORD},
    )


def csrf_headers(client: TestClient) -> dict[str, str]:
    token = client.cookies.get("tripmate_csrf")
    assert token
    return {"X-CSRF-Token": token}


def test_signup_hashes_password_and_returns_user(auth_client):
    client, session_factory = auth_client
    response = signup(client, "Alice@example.com")
    assert response.status_code == 201
    assert response.json()["user"]["email"] == "alice@example.com"
    assert response.json()["preferences"]["preferred_language"] == "en"
    with session_factory() as session:
        user = session.query(User).filter_by(email="alice@example.com").one()
        assert user.hashed_password != PASSWORD


def test_login_returns_session_and_openapi_has_account_routes(auth_client):
    client, _ = auth_client
    assert signup(client, "alice@example.com").status_code == 201
    login_client = TestClient(api.app)
    response = login_client.post(
        "/api/auth/login",
        json={"email": "alice@example.com", "password": PASSWORD},
    )
    assert response.status_code == 200
    assert login_client.get("/api/auth/me").status_code == 200
    paths = login_client.get("/openapi.json").json()["paths"]
    assert "/api/auth/signup" in paths
    assert "/api/trips/{trip_id}/resume" in paths


def test_duplicate_signup_and_invalid_login_are_rejected(auth_client):
    client, _ = auth_client
    assert signup(client, "alice@example.com").status_code == 201
    assert signup(client, "ALICE@example.com").status_code == 409
    response = client.post(
        "/api/auth/login",
        json={"email": "alice@example.com", "password": "WrongPassword123!"},
    )
    assert response.status_code == 401


def test_missing_jwt_secret_fails_before_creating_account(auth_client, monkeypatch):
    client, session_factory = auth_client
    monkeypatch.delenv("JWT_SECRET_KEY")
    response = signup(client, "alice@example.com")
    assert response.status_code == 503
    assert "JWT_SECRET_KEY" in response.json()["detail"]
    with session_factory() as session:
        assert session.query(User).filter_by(email="alice@example.com").count() == 0


def test_login_refresh_rotation_and_logout(auth_client):
    client, _ = auth_client
    assert signup(client, "alice@example.com").status_code == 201
    original_refresh = client.cookies.get(REFRESH_COOKIE)
    response = client.post("/api/auth/refresh", headers=csrf_headers(client))
    assert response.status_code == 200
    assert client.cookies.get(REFRESH_COOKIE) != original_refresh
    assert client.get("/api/auth/me").status_code == 200
    assert client.post("/api/auth/logout", headers=csrf_headers(client)).status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_expired_access_token_is_rejected(auth_client):
    client, _ = auth_client
    expired = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "sid": str(uuid.uuid4()),
            "iat": 1,
            "exp": 2,
            "type": "access",
        },
        JWT_SECRET,
        algorithm="HS256",
    )
    client.cookies.set(ACCESS_COOKIE, expired)
    assert client.get("/api/auth/me").status_code == 401


def test_profile_preferences_and_trip_crud_are_user_scoped(auth_client):
    client_a, session_factory = auth_client
    assert signup(client_a, "alice@example.com").status_code == 201
    trip_response = client_a.post(
        "/api/trips",
        headers=csrf_headers(client_a),
        json={"destination": "Kyoto", "status": "saved", "itinerary": "Visit Kyoto."},
    )
    assert trip_response.status_code == 201
    trip_id = trip_response.json()["id"]
    update = client_a.put(
        f"/api/trips/{trip_id}",
        headers=csrf_headers(client_a),
        json={"title": "Spring in Kyoto"},
    )
    assert update.status_code == 200
    assert update.json()["title"] == "Spring in Kyoto"
    assert client_a.put(
        "/api/auth/me",
        headers=csrf_headers(client_a),
        json={"name": "Alice Updated"},
    ).status_code == 200
    preferences = client_a.put(
        "/api/auth/preferences",
        headers=csrf_headers(client_a),
        json={
            "preferred_language": "hi",
            "preferred_currency": "inr",
            "travel_style": "family",
            "preferred_budget": 25000,
        },
    )
    assert preferences.status_code == 200
    assert preferences.json()["preferred_language"] == "hi"
    summary = client_a.get("/api/trips/summary")
    assert summary.status_code == 200
    assert summary.json()["total_trips"] == 1
    assert len(summary.json()["recent_trips"]) == 1

    client_b = TestClient(api.app)
    assert signup(client_b, "bob@example.com").status_code == 201
    assert client_b.get("/api/trips").json() == []
    assert client_b.get(f"/api/trips/{trip_id}").status_code == 404
    assert client_b.delete(
        f"/api/trips/{trip_id}",
        headers=csrf_headers(client_b),
    ).status_code == 404
    assert client_a.get("/api/trips").json()[0]["id"] == trip_id
    with session_factory() as session:
        trip = session.get(Trip, uuid.UUID(trip_id))
        assert trip is not None
        assert trip.user.email == "alice@example.com"
    assert client_a.delete(
        f"/api/trips/{trip_id}",
        headers=csrf_headers(client_a),
    ).status_code == 204
    assert client_a.get("/api/trips/summary").json()["total_trips"] == 0


def test_user_cannot_access_another_users_langgraph_thread(auth_client, monkeypatch):
    client_a, session_factory = auth_client
    assert signup(client_a, "alice@example.com").status_code == 201
    with session_factory() as session:
        alice = session.query(User).filter_by(email="alice@example.com").one()
        trip = Trip(
            user_id=alice.id,
            thread_id="alice-private-thread",
            destination="Kyoto",
            status="pending_approval",
        )
        session.add(trip)
        session.commit()
        alice_trip_id = str(trip.id)

    client_b = TestClient(api.app)
    assert signup(client_b, "bob@example.com").status_code == 201
    monkeypatch.setattr(
        api.travel_graph,
        "get_state",
        lambda config: pytest.fail("Graph state must not be accessed for another user's thread."),
    )
    response = client_b.post(
        "/api/travel/approve",
        headers=csrf_headers(client_b),
        json={
            "thread_id": "alice-private-thread",
            "approved": True,
            "feedback": "",
        },
    )
    assert response.status_code == 404
    resume_response = client_b.post(
        f"/api/trips/{alice_trip_id}/resume",
        headers=csrf_headers(client_b),
    )
    assert resume_response.status_code == 404


def test_owner_can_resume_their_own_pending_graph_session(auth_client, monkeypatch):
    client, session_factory = auth_client
    assert signup(client, "alice@example.com").status_code == 201
    with session_factory() as session:
        user = session.query(User).filter_by(email="alice@example.com").one()
        trip = Trip(
            user_id=user.id,
            thread_id="alice-resumable-thread",
            destination="Kyoto",
            status="pending_approval",
        )
        session.add(trip)
        session.commit()
        trip_id = str(trip.id)

    snapshot = type(
        "Snapshot",
        (),
        {"values": {"thread_id": "alice-resumable-thread"}, "next": ("human_approval",)},
    )()
    monkeypatch.setattr(api.travel_graph, "get_state", lambda config: snapshot)
    response = client.post(
        f"/api/trips/{trip_id}/resume",
        headers=csrf_headers(client),
    )
    assert response.status_code == 200
    assert response.json()["requires_approval"] is True


def test_protected_travel_endpoint_requires_authentication(auth_client):
    client, _ = auth_client
    assert client.post("/api/travel", json={"message": "Plan a trip to Kyoto"}).status_code == 401
    assert client.post(
        "/api/trips",
        json={"destination": "Kyoto"},
    ).status_code == 401
