from typing import List
from sqlalchemy import select

from src.core.config import config
from src.core.db import scoped_session
from src.sqlalchemy.models import Device, Session, Frame


class Base:
    
  def __init__(self, db: scoped_session) -> None:
    self.db = db

  def get_all_devices(self, session_id: int) -> List[Device]:
    """
    Description:
      * Fetches all the devices for a given session_id.
    
    Args:
      * session_id (int): The session_id to fetch the devices for.
    
    Returns:
      * devices (List[device]): The devices for the given session_id.
    """
    statements = (
      select(Device).where(Device.session_id == session_id)
    )
    devices = self.db.execute(statements).all()
    devices = [d[0] for d in devices]
    return devices
  
  def get_available_sessions(self) -> List[str]:
    """
    Description:
      * Fetches all the available sessions with the id and movie name.

    Returns:
      * sessions (List[str]): The list of available sessions.
    """
    statements = (
      select(Session)
    )
    sessions = self.db.execute(statements).all()
    sessions = [f"{s[0].id} - {s[0].movie[:-4]}" for s in sessions]
    
    return sessions
  
  def get_device_from_port(self, session_id: int, device_port: str) -> Device:
    """
    Description:
      * Fetches a device for a given session_id and device_port.

    Args:
      * session_id (int): The session_id to fetch the device for.
      * device_port (int): The device_port to fetch the device for.

    Returns:
      * device (Device): The device for the given session_id and device_port.
    """
    statements = (
      select(Device).where((Device.session_id == session_id) & (Device.port == device_port))
    )
    device = self.db.execute(statements).all()
    device = device[0][0]
    return device
  
  def get_sampling_rate(self, session_id: int) -> Device:
    """
    Description:
      * Fetches a session for a given session_id and device_port.

    Args:
      * session_id (int): The session_id to fetch the device for.
      * device_port (int): The device_port to fetch the device for.

    Returns:
      * device (Device): The device for the given session_id and device_port.
    """
    statements = select(Session).where(Session.id == session_id)
    session = self.db.execute(statements).all()
    session = session[0][0]
    return session.sampling_rate
  
  
  def get_eda(self, session_id: int, device_port: str) -> List[int]:
    """
    Description:
      * Fetches the eda for a given session_id and device_port.

    Args:
      * session_id (int): The session_id to fetch the eda for.
      * device_port (int): The device_port to fetch the eda for.

    Returns:
      * eda (List[int]): The eda for the given session_id and device_port.
    """
    device = self.get_device_from_port(session_id, device_port)
    statements = (
      select(Frame).where((Frame.session_id == session_id) & (Frame.device_id == device.id))
    )
    frames = self.db.execute(statements).all()
    eda = [f[0].__dict__[config.EDA_CHANNEL] for f in frames]
    x = [f[0].__dict__[config.TIME_CHANNEL] for f in frames]
    return eda, x