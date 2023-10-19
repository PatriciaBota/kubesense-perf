import time
from sqlalchemy import select

import src.sqlalchemy.models as models
from src.sqlalchemy.base import Base
from src.core.db import scoped_session


class Annotation(Base):
    
  def __init__(self, db: scoped_session) -> None:
    super().__init__(db)

  def find_or_create_annotation(self, device_id: int, session_id: int) -> models.Annotation:
    """
    Description:
      * Finds or creates an annotation for a given device_id.
    
    Args:
      * device_id (int): The device_id to find or create an annotation for.
      * session_id (int): The session_id to find or create an annotation for.
    
    Returns:
      * annotation (Annotation): The annotation for the given device_id.
    """

    # Tries to find the annotation record for the given device_id
    statement = (
      select(models.Annotation).where(
        (models.Annotation.session_id == session_id) &
        (models.Annotation.device_id == device_id))
    )
    annotation = self.db.execute(statement).all()

    # If not found, creates a new annotation record
    if annotation is None or len(annotation) < 1:
      annotation = models.Annotation(
        session_id=session_id,
        device_id=device_id,
      )

      self.db.add(annotation)
      self.db.flush()

    else:
      # The db will return a list of tuples with the first element being the annotation record
      annotation = annotation[0][0]

    return annotation
  
  def find_annotation_segment(self, device_id: int, session_id: int, start_event: int) -> models.AnnotationSegment:
    """
    Description:
      * Finds an annotation segment for a given device_id and start_event.

    Args:
      * device_id (int): The device_id to find the annotation segment for.
      * session_id (int): The session_id to find the annotation segment for.
      * start_event (int): The start_event to find the annotation segment for.

    Returns:
      * segment (AnnotationSegment): The annotation segment for the given device_id and start_event.
    """
    annotation = self.find_or_create_annotation(device_id, session_id)

    statement = (
      select(models.AnnotationSegment).where(
        (models.AnnotationSegment.annotation_id == annotation.id) &
        (models.AnnotationSegment.event_start == start_event)
      )
    )
    segment = self.db.execute(statement).all()

    if segment is None or len(segment) < 1:
      segment = models.AnnotationSegment(
      annotation_id=annotation.id,
      event_start=start_event,
      event_end=start_event + 20
      )

      self.db.add(segment)
      self.db.flush()
    else:
      # The db will return a list of tuples with the first element being the annotation record
      segment = segment[0][0]

    return segment
  
  def find_or_create_annotation_segment(self, annotation_id: int, start_event: int, end_event: int) -> models.AnnotationSegment:
    """
    Description:
      * Finds or creates an annotation segment for a given annotation_id.

    Args:
      * annotation_id (int): The annotation_id to find or create an annotation segment for.
      * start_event (int): The start_event to find or create an annotation segment for.
      * end_event (int): The end_event to find or create an annotation segment for.

    Returns:
      * segment (AnnotationSegment): The annotation segment for the given annotation_id.
    """

    statement = (
      select(models.AnnotationSegment).where(models.AnnotationSegment.annotation_id == annotation_id)
    )
    segment = self.db.execute(statement).all()
    
    segment = models.AnnotationSegment(
      annotation_id=annotation_id,
      event_start=start_event,
      event_end=end_event
    )

    self.db.add(segment)
    self.db.flush()

    return segment
  
  def upsert_annotation_record(self, type: models.AnnotationType, session_id: int, device_id: int, value: str, start_event: int) -> models.AnnotationRecord:
    """
    Description:
      * Upsert an annotation record into the database.
    
    Args:
      * type (AnnotationType): The type of annotation record to upsert.
      * session_id (int): The session_id to upsert the annotation record for.
      * device_id (int): The device_id to upsert the annotation record for.
      * value (str): The value of the annotation record to upsert.
      * start_event (int): The start_event of the annotation record to upsert.

    Returns:
      * ann_rec (AnnotationRecord): The new annotation record that was upsert into the database.
    """
    # Finds the annotation segment for the given device_id and start_event
    annotation_segment = self.find_annotation_segment(device_id, session_id, start_event)
    ann_rec = models.AnnotationRecord(
      segment_id=annotation_segment.id,
      type=type,
      value=value,
      time=int(time.time())
    )

    # Upsert the new annotation into the database
    self.db.merge(ann_rec)
    self.db.flush() 
    return ann_rec
