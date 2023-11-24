import enum
from sqlalchemy import Column, Integer,  ForeignKey, String, Float
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ENUM

from src.core.db import engine

"""
Description:
  * Initializes model tables for device and frame. These have to be an instance of Base to allow
    linking with the sqlite engine. 
"""

Base = declarative_base()

class AnnotationType(str, enum.Enum):
  AROUSAL = "AROUSAL"
  VALENCE = "VALENCE"
  AROUSAL_UNCERTAINTY = "AROUSAL_UNCERTAINTY"
  VALENCE_UNCERTAINTY = "VALENCE_UNCERTAINTY"
  TEXT = "TEXT"


class Session(Base):
  __tablename__ = "Session"
  id = Column(Integer, primary_key=True)
  device = relationship("Device", backref=backref("Device"))
  tags = relationship("AcquisitionTags", backref=backref("AcquisitionTags"))
  annotations = relationship("Annotation", backref=backref("AnnotationSession"))
  sampling_rate = Column(Integer)
  type = Column(String)
  movie = Column(String)
  start_time = Column(Float)
  end_time = Column(Float)

class AcquisitionTags(Base):
  __tablename__ = "AcquisitionTags"
  id = Column(Integer, primary_key=True)
  text = Column(String)
  elapsed = Column(Float)
  session_id = Column(Integer, ForeignKey("Session.id"))

class Device(Base):
  __tablename__ = "Device"
  id = Column(Integer, primary_key=True)
  session_id = Column(Integer, ForeignKey("Session.id"))
  port = Column(String, index=True)
  start_time = Column(Float)
  frames = relationship("Frame", backref=backref("Frame"))
  annotations = relationship("Annotation", backref=backref("AnnotationDevice"))

class Frame(Base):
  __tablename__ = "Frame"
  id = Column(Integer, primary_key=True)
  device_id = Column(Integer, ForeignKey("Device.id"))
  session_id = Column(Integer, index=True)  # Mostly used for control
  seq = Column(Integer)
  ai_1 = Column(Integer)
  ai_2 = Column(Integer)
  ai_3 = Column(Integer)
  ai_4 = Column(Integer)
  ai_5 = Column(Integer)
  ai_6 = Column(Integer)
  timestamp = Column(Integer, index=True)

class Annotation(Base):
  __tablename__ = "Annotation"
  id = Column(Integer, primary_key=True)
  session_id = Column(Integer, ForeignKey("Session.id"))
  device_id = Column(Integer, ForeignKey("Device.id"))
  segments = relationship("AnnotationSegment", backref=backref("AnnotationSegment"))

class AnnotationSegment(Base):
  __tablename__ = "AnnotationSegment"
  id = Column(Integer, primary_key=True)
  annotation_id = Column(Integer, ForeignKey("Annotation.id"))
  records = relationship("AnnotationRecord", backref=backref("AnnotationRecord"))
  event_start = Column(Integer)
  event_end = Column(Integer)

class AnnotationRecord(Base):
  __tablename__ = "AnnotationRecord"
  id = Column(Integer, primary_key=True)
  segment_id = Column(Integer, ForeignKey("AnnotationSegment.id"))
  time = Column(Integer)
  value = Column(String)
  type = Column(ENUM(AnnotationType, name="annotation_type_enum", native_enum=True))



Base.metadata.create_all(engine)
