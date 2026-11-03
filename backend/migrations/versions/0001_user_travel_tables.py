"""Create user, preference, trip, conversation, and auth session tables."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_user_travel_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("hashed_password", sa.String(512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "user_preferences",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("preferred_language", sa.String(16), nullable=False, server_default="en"),
        sa.Column("preferred_currency", sa.String(8), nullable=False, server_default="INR"),
        sa.Column("travel_style", sa.String(40), nullable=False, server_default=""),
        sa.Column("accommodation_preferences", sa.Text(), nullable=False, server_default=""),
        sa.Column("preferred_budget", sa.Numeric(14, 2)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", name="uq_user_preferences_user_id"),
    )
    op.create_index("ix_user_preferences_user_id", "user_preferences", ["user_id"])

    op.create_table(
        "trips",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("thread_id", sa.String(128)),
        sa.Column("title", sa.String(200), nullable=False, server_default=""),
        sa.Column("origin", sa.String(160), nullable=False, server_default=""),
        sa.Column("destination", sa.String(160), nullable=False),
        sa.Column("travel_start_date", sa.Date()),
        sa.Column("travel_end_date", sa.Date()),
        sa.Column("budget", sa.Numeric(14, 2)),
        sa.Column("status", sa.String(40), nullable=False, server_default="pending_approval"),
        sa.Column("itinerary", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("thread_id", name="uq_trips_thread_id"),
        sa.UniqueConstraint("id", "user_id", name="uq_trips_id_user_id"),
    )
    op.create_index("ix_trips_user_created", "trips", ["user_id", "created_at"])
    op.create_index("ix_trips_user_destination", "trips", ["user_id", "destination"])
    op.create_index("ix_trips_user_status", "trips", ["user_id", "status"])

    op.create_table(
        "trip_conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "trip_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["trip_id", "user_id"],
            ["trips.id", "trips.user_id"],
            ondelete="CASCADE",
            name="fk_trip_conversations_owned_trip",
        ),
    )
    op.create_index(
        "ix_trip_conversations_trip_created",
        "trip_conversations",
        ["trip_id", "created_at"],
    )
    op.create_index(
        "ix_trip_conversations_user_created",
        "trip_conversations",
        ["user_id", "created_at"],
    )

    op.create_table(
        "auth_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("refresh_token_hash", sa.String(64), nullable=False),
        sa.Column("csrf_token_hash", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("refresh_token_hash", name="uq_auth_sessions_refresh_token_hash"),
    )
    op.create_index(
        "ix_auth_sessions_user_expires",
        "auth_sessions",
        ["user_id", "expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_auth_sessions_user_expires", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_index("ix_trip_conversations_user_created", table_name="trip_conversations")
    op.drop_index("ix_trip_conversations_trip_created", table_name="trip_conversations")
    op.drop_table("trip_conversations")
    op.drop_index("ix_trips_user_status", table_name="trips")
    op.drop_index("ix_trips_user_destination", table_name="trips")
    op.drop_index("ix_trips_user_created", table_name="trips")
    op.drop_table("trips")
    op.drop_index("ix_user_preferences_user_id", table_name="user_preferences")
    op.drop_table("user_preferences")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
