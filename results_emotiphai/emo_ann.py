from typing import List, Generator
import multiprocessing as prc
import numpy as np
import math
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker, scoped_session, Session
import pdb
import matplotlib.pyplot as plt
from sqlalchemy import distinct
import pandas as pd

from src.sqlalchemy.models import Session as ModelSession, Device, Frame, AnnotationRecord, AnnotationSegment, Annotation
from sqlalchemy import create_engine


class DataAnalysis:

    def __init__(self, db_url: str, DIMENSION: str) -> None:
        self.database: str = db_url
        self.engine = create_engine(self.database, pool_pre_ping=True)
        self.dimension: str = DIMENSION

    def _init_db_session(self) -> Session:
        """
        Description:
            * Builds database engine and starts connection to the database.

        Returns:
            * database session
        """
        session = scoped_session(sessionmaker(bind=self.engine))
        return session()
    
    def get_ann(self, log: bool = False) -> None:
        """
        Description:
        * Calculates data loss for the entire acquisition session.
        """
        session: Session = self._init_db_session()

        statement = session.query(ModelSession.id).distinct().all()
        session_ids = [row[0] for row in statement]
        results_table: List = []

        for session_id in session_ids:
            print(f"Session id: {session_id}")
            devices: List[Device] = session.query(Device).filter(Device.session_id == session_id).all()
            session_obj = session.query(ModelSession).filter(ModelSession.id == session_id).one()

            sampling_rate = session_obj.sampling_rate

            for device in devices:

                try:
                    start_time = device.start_time
                    end_time = session_obj.end_time
                    exp_duration = end_time - start_time
                except Exception as e:
                    print(e)
                    exp_duration = 0

                # Gets all device frames and splits the output into sequence numbers and timestamps
                statement = (
                    select(Annotation.id)
                    .join(Device)
                    .where(Annotation.id == device.id, Device.session_id == session_id)
                )
                annotation_id = session.execute(statement).all()[0][0]
                
                statement = (
                    select(AnnotationSegment.id)
                    .where(AnnotationSegment.annotation_id == annotation_id)
                )
                ann_seg_id = session.execute(statement).all()
                for ann_seg in ann_seg_id:
                    ann_seg = ann_seg[0]
                    statement = (
                        select(AnnotationRecord.value)
                        .where((AnnotationRecord.segment_id == ann_seg) & (AnnotationRecord.type == self.dimension))
                    )
                    sc_ann_record = session.execute(statement).all()
                    for row in sc_ann_record:
                        sc_ann_record = row[0]
                    
                    print(f"Device id: {device.id} | Annotation id {annotation_id} | Annotation segment id: {ann_seg} | {DIMENSION}: {sc_ann_record}")
                    results_table.append([session_id, device.port, sc_ann_record])
                    

        session.close()
        return results_table
    
db_path = "sqlite:///data/db.sqlite"  # This is the correct format for an SQLite URL  # Replace with your actual database path
DIMENSION = "AROUSAL"  # Replace with your actual dimension (AROUSAL or UNCERTAINTY_VALENCE)
data_analysis = DataAnalysis(db_url=db_path, DIMENSION=DIMENSION)

# Now, call the calc_data_loss method. You can pass the session_id and log flag if needed.
results_table = data_analysis.get_ann(log=False) 

df = pd.DataFrame(results_table, columns=["session_id", "device_id", "sc_ann_record"])
averages = df.mean(numeric_only=True)  # get the average of each numeric column
std_devs = df.std(numeric_only=True)  # get the standard deviation of each numeric column

summary_row = ['---' if col not in averages.index else f"{averages[col]:.2f} ± {std_devs[col]:.2f}" for col in df.columns]
df.loc['Average'] = summary_row

df.to_csv("results/ann.csv", index=False)
pdb.set_trace()