from typing import List
from sqlalchemy import select

from src.sqlalchemy.base import Base
from src.core.db import scoped_session
from src.sqlalchemy.models import Frame, AcquisitionTags
from sqlalchemy.sql.expression import and_

class Acquisition(Base):
    
  def __init__(self, db: scoped_session) -> None:
    super().__init__(db)

  def save_tag(self, text: str, elapsed: int, session_id: int) -> None:
    """
    Description:
      * Saves a tag to the database.
    
    Args:
      * text (str): The text of the tag.
      * elapsed (int): The elapsed time of the tag.
      * session_id (int): The session_id of the tag.
    """
    tag = AcquisitionTags(
      text=text,
      elapsed=elapsed,
      session_id=session_id,
    )
    self.db.add(tag)
    self.db.flush()

  def fetch_frames(self, session_id: int, device_id: int, limit: dict[str, int]) -> List[Frame]:
    """
    Description:
      * Fetches the frames for a given session_id.

    Args:
      * session_id (int): The session_id to fetch the frames for.
      * device_id (int): The device_id to fetch the frames for.
      * limit (int): The number of rows to fetch.
    
    Returns:
      * frames (List[Frame]): The frames for the given session_id.
    """
    # Start building the query
    query = select(Frame)
    # List of conditions
    conditions = []
    # Add each condition only if the value exists
    if session_id is not None:
      conditions.append(Frame.session_id == session_id)
    if device_id is not None:
      conditions.append(Frame.device_id == device_id)
    if str(device_id) in limit.keys(): 
      conditions.append(Frame.timestamp > int(limit[str(device_id)]))

    # Add the conditions to the query
    query = query.where(and_(*conditions))

    # Add ordering and (potentially) limit
    statement = query.order_by(Frame.id.desc())

    frames = reversed(self.db.execute(statement).all())
    frames = [f[0] for f in frames]
    return frames
