from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from os import getenv

DATABASE_URL: str = f"sqlite://{getenv('DATABASE_PATH', '/data/db.sqlite')}"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

session_not_thread_safe = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# This is going to create a scoped thread safe database session
# Source: https://docs.sqlalchemy.org/en/20/orm/contextual.html
SessionLocal = scoped_session(session_factory=session_not_thread_safe)
