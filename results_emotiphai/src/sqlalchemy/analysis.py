from src.sqlalchemy.base import Base
from src.core.db import scoped_session


class Analysis(Base):
    
  def __init__(self, db: scoped_session) -> None:
    super().__init__(db)
