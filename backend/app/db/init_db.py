from __future__ import annotations

from sqlalchemy import text

from app.db.session import engine
from app.models import Base


def init_db() -> None:
    """Initialize database tables without relying on Alembic migrations."""
    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        # Enable foreign key enforcement for SQLite if applicable
        conn.execute(text("PRAGMA foreign_keys=ON"))


if __name__ == "__main__":
    init_db()
