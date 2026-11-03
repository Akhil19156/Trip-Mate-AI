import secrets

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .auth import (
    CSRF_COOKIE,
    REFRESH_COOKIE,
    authenticate_refresh_session,
    authenticate_user,
    clear_auth_cookies,
    create_auth_session,
    ensure_jwt_secret,
    get_current_user,
    hash_password,
    require_csrf,
    revoke_session,
    set_session_cookies,
)
from .database import get_db
from .models import AuthSession, User, UserPreference
from .schemas_auth import (
    LoginRequest,
    PreferencesResponse,
    PreferencesUpdate,
    ProfileUpdate,
    SignupRequest,
    UserResponse,
)
from .trip_service import ensure_user_preferences

router = APIRouter(prefix="/api/auth", tags=["authentication"])


@router.post("/signup", status_code=201)
def signup(
    payload: SignupRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    ensure_jwt_secret()
    user = User(
        name=payload.name.strip(),
        email=str(payload.email).strip().casefold(),
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    try:
        db.flush()
        user.preferences = UserPreference(user_id=user.id)
        auth_session, refresh_token, csrf_token = create_auth_session(user.id)
        db.add(auth_session)
        db.commit()
        db.refresh(user)
        db.refresh(auth_session)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="An account with that email already exists.",
        ) from exc
    set_session_cookies(response, user.id, auth_session, refresh_token, csrf_token)
    return {
        "user": UserResponse.model_validate(user),
        "preferences": PreferencesResponse.model_validate(user.preferences),
    }


@router.post("/login")
def login(
    payload: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    ensure_jwt_secret()
    user = authenticate_user(db, str(payload.email), payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    preferences = ensure_user_preferences(db, user)
    auth_session, refresh_token, csrf_token = create_auth_session(user.id)
    db.add(auth_session)
    db.commit()
    db.refresh(auth_session)
    set_session_cookies(response, user.id, auth_session, refresh_token, csrf_token)
    return {
        "user": UserResponse.model_validate(user),
        "preferences": PreferencesResponse.model_validate(preferences),
    }


@router.post("/refresh")
def refresh(
    response: Response,
    db: Session = Depends(get_db),
    refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE),
    csrf_cookie: str | None = Cookie(default=None, alias=CSRF_COOKIE),
    csrf_header: str | None = Header(default=None, alias="X-CSRF-Token"),
):
    ensure_jwt_secret()
    if (
        not refresh_token
        or not csrf_cookie
        or not csrf_header
        or not secrets.compare_digest(csrf_cookie, csrf_header)
    ):
        raise HTTPException(status_code=401, detail="A valid refresh session is required.")
    old_session = authenticate_refresh_session(db, refresh_token, csrf_cookie)
    if old_session is None:
        clear_auth_cookies(response)
        raise HTTPException(status_code=401, detail="A valid refresh session is required.")
    user_id = old_session.user_id
    revoke_session(old_session)
    new_session, new_refresh, new_csrf = create_auth_session(user_id)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    set_session_cookies(response, user_id, new_session, new_refresh, new_csrf)
    return {"success": True}


@router.post("/logout")
def logout(
    response: Response,
    user: User = Depends(require_csrf),
    db: Session = Depends(get_db),
):
    sessions = db.scalars(
        select(AuthSession).where(
            AuthSession.user_id == user.id,
            AuthSession.revoked_at.is_(None),
        )
    )
    for auth_session in sessions:
        revoke_session(auth_session)
    db.commit()
    clear_auth_cookies(response)
    return {"success": True}


@router.get("/me")
def me(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    preferences = ensure_user_preferences(db, user)
    db.commit()
    return {
        "user": UserResponse.model_validate(user),
        "preferences": PreferencesResponse.model_validate(preferences),
    }


@router.put("/me")
def update_profile(
    payload: ProfileUpdate,
    user: User = Depends(require_csrf),
    db: Session = Depends(get_db),
):
    user.name = payload.name.strip()
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@router.put("/preferences")
def update_preferences(
    payload: PreferencesUpdate,
    user: User = Depends(require_csrf),
    db: Session = Depends(get_db),
):
    preferences = ensure_user_preferences(db, user)
    for field, value in payload.model_dump().items():
        setattr(preferences, field, value)
    db.commit()
    db.refresh(preferences)
    return PreferencesResponse.model_validate(preferences)
