from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

class EdaData(BaseModel):
  device_id: int
  ch_1: Optional[List[float]] = None
  ch_2: Optional[List[float]] = None
  ch_3: Optional[List[float]] = None
  ch_4: Optional[List[float]] = None
  ch_5: Optional[List[float]] = None
  ch_6: Optional[List[float]] = None
  db_device_id: int
  ts: int

class AppMode(str, Enum):
  start: str = "start"
  stop: str = "stop"
