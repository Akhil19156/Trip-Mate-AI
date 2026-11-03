import hashlib
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import Cookie, Depends, Header, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import get_db
from .models import AuthSession, User

ACCESS_COOKIE = "tripmate_access"
REFRESH_COOKIE = "tripmate_refresh"
CSRF_COOKIE = "tripmate_csrf"
ACCESS_TTL = timedelta(minutes=15)
REFRESH_TTL = timedelta(days=14)
_password_hasher = PasswordHasher()
_dummy_password_hash = _password_hasher.hash(secrets.token_urlsafe(24))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _is_expired(value: datetime) -> bool:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value <= _utcnow()


def _secret_key() -> str:
    secret = os.getenv("JWT_SECRET_KEY", "")
    if len(secret.encode("utf-8")) < 32:
        raise RuntimeError("JWT_SECRET_KEY must contain at least 32 bytes.")
    return secret


def ensure_jwt_secret() -> None:
    try:
        _secret_key()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Authentication is not configured. Set JWT_SECRET_KEY to a "
                "random value of at least 32 bytes and restart the backend."
            ),
        ) from exc


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    try:
        return _password_hasher.verify(hashed_password, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def _cookie_settings() -> dict[str, object]:
    same_site = os.getenv("AUTH_COOKIE_SAMESITE", "lax").lower()
    if same_site not in {"lax", "strict", "none"}:
        same_site = "lax"
    secure = os.getenv("AUTH_COOKIE_SECURE", "").lower()
    secure_cookie = (
        secure == "true"
        if secure in {"true", "false"}
        else os.getenv("APP_ENV", "development").lower() == "production"
    )
    if same_site == "none":
        secure_cookie = True
    return {
        "secure": secure_cookie,
        "samesite": same_site,
        "path": "/",
    }


def create_auth_session(user_id: uuid.UUID) -> tuple[AuthSession, str, str]:
    refresh_token = secrets.token_urlsafe(48)
    csrf_token = secrets.token_urlsafe(32)
    auth_session = AuthSession(
        user_id=user_id,
        refresh_token_hash=hash_token(refresh_token),
        csrf_token_hash=hash_token(csrf_token),
        expires_at=_utcnow() + REFRESH_TTL,
    )
    return auth_session, refresh_token, csrf_token


def _access_token(user_id: uuid.UUID, session_id: uuid.UUID) -> str:
    now = _utcnow()
    return jwt.encode(
        {
            "sub": str(user_id),
            "sid": str(session_id),
            "iat": now,
            "exp": now + ACCESS_TTL,
            "type": "access",
        },
        _secret_key(),
        algorithm="HS256",
    )


def set_session_cookies(
    response: Response,
    user_id: uuid.UUID,
    session: AuthSession,
    refresh_token: str,
    csrf_token: str,
) -> None:
    settings = _cookie_settings()
    response.set_cookie(
        ACCESS_COOKIE,
        _access_token(user_id, session.id),
        max_age=int(ACCESS_TTL.total_seconds()),
        httponly=True,
        **settings,
    )
    response.set_cookie(
        REFRESH_COOKIE,
        refresh_token,
        max_age=int(REFRESH_TTL.total_seconds()),
        httponly=True,
        **settings,
    )
    response.set_cookie(
        CSRF_COOKIE,
        csrf_token,
        max_age=int(REFRESH_TTL.total_seconds()),
        httponly=False,
        **settings,
    )


def clear_auth_cookies(response: Response) -> None:
    settings = _cookie_settings()
    for name in (ACCESS_COOKIE, REFRESH_COOKIE, CSRF_COOKIE):
        response.delete_cookie(name, **settings)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Authentication is required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _decode_access_token(token: str) -> tuple[uuid.UUID, uuid.UUID]:
    try:
        claims = jwt.decode(
            token,
            _secret_key(),
            algorithms=["HS256"],
            options={"require": ["sub", "sid", "exp", "iat", "type"]},
        )
        if claims.get("type") != "access":
            raise _unauthorized()
        return uuid.UUID(claims["sub"]), uuid.UUID(claims["sid"])
    except HTTPException:
        raise
    except (jwt.PyJWTError, ValueError, RuntimeError) as exc:
        raise _unauthorized() from exc


def get_current_user(
    access_token: Annotated[str | None, Cookie(alias=ACCESS_COOKIE)] = None,
    db: Session = Depends(get_db),
) -> User:
    if not access_token:
        raise _unauthorized()
    user_id, session_id = _decode_access_token(access_token)
    auth_session = db.get(AuthSession, session_id)
    if (
        auth_session is None
        or auth_session.user_id != user_id
        or auth_session.revoked_at is not None
        or _is_expired(auth_session.expires_at)
    ):
        raise _unauthorized()
    user = db.get(User, user_id)
    if user is None:
        raise _unauthorized()
    return user


def require_csrf(
    user: User = Depends(get_current_user),
    csrf_header: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
    csrf_cookie: Annotated[str | None, Cookie(alias=CSRF_COOKIE)] = None,
    access_token: Annotated[str | None, Cookie(alias=ACCESS_COOKIE)] = None,
    db: Session = Depends(get_db),
) -> User:
    if not access_token or not csrf_header or not csrf_cookie:
        raise HTTPException(status_code=403, detail="CSRF validation failed.")
    if not secrets.compare_digest(csrf_header, csrf_cookie):
        raise HTTPException(status_code=403, detail="CSRF validation failed.")
    _, session_id = _decode_access_token(access_token)
    auth_session = db.get(AuthSession, session_id)
    if (
        auth_session is None
        or auth_session.user_id != user.id
        or not secrets.compare_digest(
            auth_session.csrf_token_hash,
            hash_token(csrf_header),
        )
    ):
        raise HTTPException(status_code=403, detail="CSRF validation failed.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
ProtectedUser = Annotated[User, Depends(require_csrf)]


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    normalized_email = email.strip().casefold()
    user = db.scalar(select(User).where(User.email == normalized_email))
    hashed_password = user.hashed_password if user else _dummy_password_hash
    valid = verify_password(password, hashed_password)
    return user if user is not None and valid else None


def authenticate_refresh_session(
    db: Session,
    refresh_token: str,
    csrf_token: str,
) -> AuthSession | None:
    auth_session = db.scalar(
        select(AuthSession)
        .where(AuthSession.refresh_token_hash == hash_token(refresh_token))
        .with_for_update()
    )
    if (
        auth_session is None
        or auth_session.revoked_at is not None
        or _is_expired(auth_session.expires_at)
        or not secrets.compare_digest(
            auth_session.csrf_token_hash,
            hash_token(csrf_token),
        )
    ):
        return None
    return auth_session


def revoke_session(auth_session: AuthSession) -> None:
    auth_session.revoked_at = _utcnow()
