from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings


# Create the SQLAlchemy engine using the existing database URL.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
)


# Factory for creating database sessions.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)