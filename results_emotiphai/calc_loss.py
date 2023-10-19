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

from src.sqlalchemy.models import Session as ModelSession, Device, Frame
from sqlalchemy import create_engine


class DataAnalysis:

    def __init__(self, db_url: str) -> None:
        self.database: str = db_url
        self.engine = create_engine(self.database, pool_pre_ping=True)

    def _init_db_session(self) -> Session:
        """
        Description:
            * Builds database engine and starts connection to the database.

        Returns:
            * database session
        """
        session = scoped_session(sessionmaker(bind=self.engine))
        return session()
    
    def seconds_to_hours_minutes_seconds(self, time_in_seconds):
        """
        Convert a time duration from seconds to hours, minutes, and seconds.

        Parameters:
        time_in_seconds (int or float): The total time in seconds.

        Returns:
        tuple: A tuple containing time in hours, minutes, and seconds.
        """

        # Calculate hours, minutes, and seconds
        hours, remainder = divmod(time_in_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return hours, minutes, seconds
    
    def calculate_data_collection_time(self, sampling_rate, number_of_samples):
        """
        Calculate the total data collection time.

        Parameters:
        sampling_rate (int or float): The number of samples per second.
        number_of_samples (int): The total number of samples collected.

        Returns:
        float: The total data collection time in seconds.
        """

        # Total time is the number of samples divided by the rate (samples per second)
        # which gives the total time in seconds.
        total_time_seconds = number_of_samples / float(sampling_rate)

        return total_time_seconds
    

    def calc_data_loss(self, log: bool = False) -> None:
        """
        Description:
        * Calculates data loss for the entire acquisition session.
        """
        session: Session = self._init_db_session()

        max_seq_number = 4095
        statement = session.query(ModelSession.id).distinct().all()
        session_ids = [row[0] for row in statement]

        results_table: List = []

        for session_id in session_ids:
            print(f"Session id: {session_id}")
            devices: List[Device] = session.query(Device).filter(Device.session_id == session_id).all()
            sampling_rate: int = session.query(ModelSession).filter(ModelSession.id == session_id).one().sampling_rate

            for device in devices:

                n_duplicates: int = 0
                n_missed_packets: int = 0
                breaks: int = 0
                total_break_time: int = 0
                avg_break_time: List = []

                # Gets all device frames and splits the output into sequence numbers and timestamps
                statement = (
                    select(Frame.seq, Frame.timestamp)
                    .join(Device)
                    .where(Frame.device_id == device.id, Device.session_id == session_id)
                    .order_by(Frame.timestamp)
                )
                frames: List[Frame] = session.execute(statement).all()
                sequences, timestamps = zip(*frames)
                timestamps = np.array(timestamps)
                #plt.figure()
                #plt.title(device.port)
                #plt.plot(timestamps - timestamps[0], ".")
                ##plt.plot(sequences, ".")
                #plt.show()

                copy_frames = frames[1:]

                # Applies numpy diff to sequence numbers to check if there are any missing packages
                differences: List[int] = np.diff(sequences).tolist()

                # Goes over all differences and corresponding timestamps and counts data loss
                print(f"unique sequences: {len(np.unique(sequences))}")
                for i, diff in enumerate(differences):

                    # Counts number of frames based on the elapsed time and counts number of possible full loops
                    n_frames_passed = timestamps[i + 1] - timestamps[i] // sampling_rate
                    n_full_loops = 0 if n_frames_passed == 0 else max_seq_number // n_frames_passed

                    if log:
                        print(f"INFO: index: {i} | diff: {diff} | n_full_loops: {n_full_loops} | missed_packets: {n_missed_packets} | frame: {copy_frames[i]}")

                    if diff != 1 and diff != -max_seq_number:  # If is not one, then we jumped more than one frame and so, we have an error

                        if diff == 0:  # Has a duplicate if the timestamp is the same or is a full round loss
                            if timestamps[i] == timestamps[i + 1]:
                                n_duplicates += 1

                            elif n_full_loops > 0:
                                if log:
                                    print(f"ERROR: got full loop and added: {n_full_loops * max_seq_number}")
                                missed = max_seq_number * n_full_loops  # We lost an entire set of frames
                                n_missed_packets += missed

                        elif diff > 1:  # We missed some packets but it did not go back to sequence 0
                            if log:
                                print(f"ERROR: Missed some packets: {diff - 1}")
                            missed = diff - 1
                            n_missed_packets += missed
                            breaks += 1
                            total_break_time += diff / sampling_rate
                            avg_break_time += [ diff / sampling_rate]
                        
                        elif diff < 0:  # We came back to sequence 0 (i.e: (15 - 12) + (7 - 0))
                            if log:
                                print(f"ERROR: Missed packets in break: {(max_seq_number - sequences[i - 1]) + (sequences[i] - 0)}")
                            missed = (max_seq_number - sequences[i - 1]) + (sequences[i] - 0)
                            n_missed_packets += missed
                            breaks += 1
                            total_break_time += missed / sampling_rate
                            avg_break_time += [ missed / sampling_rate]

                total_number_of_packets = len(sequences)
                collection_time = self.calculate_data_collection_time(sampling_rate, len(sequences))
                hours, minutes, seconds = self.seconds_to_hours_minutes_seconds(collection_time)
                if len(avg_break_time) == 0:
                    avg_break_time = [0]
                if n_missed_packets == 0:
                    assert np.unique(differences).shape[0] == 2
                results_table.append([session_id, device.port, round(n_missed_packets / total_number_of_packets * 100, 2),  # rounds to 2 decimal places
                round(n_duplicates / total_number_of_packets * 100, 2), f"{hours}:{minutes}:{seconds}", sampling_rate, 
                breaks, round(total_break_time, 2), f"{round(np.mean(avg_break_time), 2)} +- {round(np.std(avg_break_time), 2)}"])
                print(f"SESSION: {session_id}; [Device = {device.port}] -> Missed packets = {n_missed_packets}, {(n_missed_packets / total_number_of_packets * 100):.2f}% | Duplicates = {(n_duplicates / total_number_of_packets * 100):.2f}% Duration {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds | Breaks = {breaks} | Break time = {total_break_time:.2f} seconds | Avg break time = {np.mean(avg_break_time):.2f} +- {np.std(avg_break_time):.2f} seconds")

            print("%%%%% SESSION END %%%%% \n")
        session.close()
        return results_table

    
db_path = "sqlite:///data/db.sqlite"  # This is the correct format for an SQLite URL  # Replace with your actual database path
data_analysis = DataAnalysis(db_url=db_path)

# Now, call the calc_data_loss method. You can pass the session_id and log flag if needed.
results_table = data_analysis.calc_data_loss(log=False) 

df = pd.DataFrame.from_dict(results_table) 
header = ['session_id', 'device', 'data_loss (%)', 'duplicates (%)', 'duration (h:m:s)', 'sampling_rate (Hz)', 'breaks', 'total break_time (s)', 'avg break time (s)']
averages = df.mean(numeric_only=True)  # get the average of each numeric column
std_devs = df.std(numeric_only=True)  # get the standard deviation of each numeric column

summary_row = ['---' if col not in averages.index else f"{averages[col]:.2f} ± {std_devs[col]:.2f}" for col in df.columns]
df.loc['Average'] = summary_row
df.to_csv ('results/db_results.csv', index = True, header=header)
