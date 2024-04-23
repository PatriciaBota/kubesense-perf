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

import pdb

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
            session_obj = session.query(ModelSession).filter(ModelSession.id == session_id).one()

            sampling_rate = session_obj.sampling_rate

            print(f"sampling_rate {sampling_rate}")

            for device in devices:

                n_duplicates: int = 0
                n_missed_packets: int = 0
                seq_rest: int = 0
                loss_points: int = 0
                total_loss_time: int = 0
                avg_break_time: List = []
                buffer_full_event = 0
                buffer_full_ts = []
                time_resets_count: int = 0
                time_resets_ts = []
                buffer_lost_frames: int = 0
                seq_loss: int = 0
                total_pack_seq: int = 0
                seq_loss_ts: List = []
                buffer_full_idx: List = []
                seq_loss_idx: List = []
                
                try:
                    start_time = device.start_time
                    end_time = session_obj.end_time
                    exp_duration = end_time - start_time
                except Exception as e:
                    print(e)
                    exp_duration = 0

                # Gets all device frames and splits the output into sequence numbers and timestamps
                statement = (
                    select(Frame.seq, Frame.timestamp, Frame.ai_1, Frame.ai_3)
                    .join(Device)
                    .where(Frame.device_id == device.id, Device.session_id == session_id)
                    .order_by(Frame.timestamp)
                )
                frames: List[Frame] = session.execute(statement).all()
                sequences, timestamps, eda, ppg = zip(*frames)
                total_number_of_packets = len(sequences)
                total_pack_seq = len(sequences)
                timestamps = np.array(timestamps) * 10**-6 # to seconds
                
                # Applies numpy diff to sequence numbers to check if there are any missing packages
                differences: List[int] = np.diff(sequences).tolist()
                sampling_period: List[int] = np.diff(timestamps).tolist()
                idx_neg = np.where(np.array(sampling_period) < 0)[0]


                if log:
                    if len(idx_neg) > 0:
                        plt.figure()
                        plt.title(str(device.port) + "neg sp")
                        for index in idx_neg:
                            plt.axvline(x=index, color='r', linestyle='--')  # Plots a vertical line at each index
                        #plt.plot(timestamps, ".")
                        plt.plot(timestamps, ".")
                        plt.ylabel("timestamp (s)")
                        plt.show()
                
                # Goes over all differences and corresponding timestamps and counts data loss
                #print(f"unique sequences: {len(np.unique(sequences))}, total size", {len(sequences)})
                for i, diff in enumerate(differences):

                    # Counts number of frames based on the elapsed time and counts number of possible full loops
                    if sampling_period[i] > 0:
                        n_frames_passed = math.floor((timestamps[i + 1] - timestamps[i]) * sampling_rate)
                    else:
                        n_frames_passed = 0
                    n_full_loops = 0 if n_frames_passed == 1 else n_frames_passed // max_seq_number 

                    #if log:
                    #    print(f"INFO: index: {i} | diff: {diff} | n_full_loops: {n_full_loops} | missed_packets: {n_missed_packets} | frame: {copy_frames[i]}")

                    if diff != 1 and diff != -max_seq_number:  # If is not one, then we jumped more than one frame and so, we have an error

                        if diff == 0:  # Has a duplicate if the timestamp is the same or is a full round loss
                            if len(np.unique(timestamps[i])) > 1:
                                n_duplicates += 1

                            elif n_full_loops > 0:
                                if log:
                                    print(f"ERROR: got full loop and added: {n_full_loops * max_seq_number}")
                                missed = max_seq_number * n_full_loops + (max_seq_number - sequences[i]) + (sequences[i+1] - 0)   # We lost an entire set of frames
                                n_missed_packets += missed
                                total_number_of_packets += missed
                                loss_points += 1
                                if sampling_period[i] > 0:
                                    total_loss_time += timestamps[i+1] - timestamps[i] #missed / sampling_rate
                                    avg_break_time += [timestamps[i+1] - timestamps[i]]
                                    seq_loss_ts += [timestamps[i+1] - timestamps[i]]
                                else:
                                    total_loss_time += missed / sampling_rate # 1 4: 2, 3 -> 0.01*2
                                    avg_break_time += [missed / sampling_rate]
                                    seq_loss_ts += [missed / sampling_rate]
                                seq_loss += missed
                                total_pack_seq += missed
                                seq_loss_idx += [i+1]

                        elif diff > 1:  # We missed some packets but it did not go back to sequence 0
                            if log:
                                print(f"ERROR: Missed some packets: {diff - 1}") 
                            missed = diff - 1
                            n_missed_packets += missed
                            loss_points += 1
                            if sampling_period[i] > 0:
                                total_loss_time += timestamps[i+1] - timestamps[i] 
                                avg_break_time += [timestamps[i+1] - timestamps[i]]
                                seq_loss_ts += [timestamps[i+1] - timestamps[i]]
                            else:
                                total_loss_time += missed / sampling_rate
                                avg_break_time += [ missed / sampling_rate]
                                seq_loss_ts += [missed / sampling_rate]
                            total_number_of_packets += missed
                            seq_loss += missed
                            total_pack_seq += missed
                            seq_loss_idx += [i+1]
                        
                        elif diff < 0:  
                            if sequences[i+1] == 0 and sequences[i] == max_seq_number:  # device turn off, don't know for how long
                                seq_rest += 1 # break with no data loss, reinicia seq number
                            else:  # We came back to sequence 0 (i.e: (15 - 12) + (7 - 0))
                                seq_rest += 1 # break with no data loss, reinicia seq number
                                if log:
                                    print(f"ERROR: Missed packets in break: {(max_seq_number - sequences[i]) + (sequences[i+1] - 0)}")
                                missed = (max_seq_number - sequences[i]) + (sequences[i+1] - 0)
                                n_missed_packets += missed
                                loss_points += 1
                                total_loss_time += missed / sampling_rate
                                avg_break_time += [ missed / sampling_rate]
                                total_number_of_packets += missed 
                                seq_loss += missed
                                total_pack_seq += missed
                                seq_loss_ts += [missed / sampling_rate]
                                seq_loss_idx += [i+1]
                    else:
                        if sampling_period[i] > 2*(1/sampling_rate): # there was loss without change in paket number
                            # can not send or receive data at high speed enough, buffer becomes full -> aquisition stops -> data loss not in transmission but in aquisition
                            if diff == 1:
                                buffer_full_event += 1 # max # min, não recolhe dados por ter buffer cheio
                                buffer_full_ts += [timestamps[i+1] - timestamps[i]] # max # min, não recolhe dados por ter buffer cheio
                                missed = math.floor((timestamps[i + 1] - timestamps[i]) * sampling_rate)
                                n_missed_packets += missed
                                loss_points += 1
                                total_loss_time += timestamps[i + 1] - timestamps[i]
                                avg_break_time += [timestamps[i + 1] - timestamps[i]]
                                total_number_of_packets += missed  
                                buffer_lost_frames += missed
                                buffer_full_idx += [i+1]
                if log:
                    if len(buffer_full_idx) > 0:
                        plt.figure()
                        plt.title(str(device.port) + " buffer full")
                        for index in buffer_full_idx:
                            plt.axvline(x=index, color='r', linestyle='--')  # Plots a vertical line at each index
                        #plt.plot(timestamps, ".")
                        plt.plot(timestamps, ".")
                        plt.ylabel("timestamps")
                        plt.show()

                sampling_period = np.array(sampling_period)
                time_resets = np.where(sampling_period < 0)[0] +1
                time_resets_count = len(time_resets)
                time_resets_ts = [sampling_period[i] for i in time_resets]
                
                sampling_period = [num for num in sampling_period if num >= 0]

                collection_time = self.calculate_data_collection_time(sampling_rate, len(sequences))
                hours, minutes, seconds = self.seconds_to_hours_minutes_seconds(collection_time)
                exp_hours, exp_minutes, exp_seconds = self.seconds_to_hours_minutes_seconds(exp_duration)
                if len(avg_break_time) == 0:
                    avg_break_time = [0]
                if n_missed_packets == 0:
                    assert (np.unique(differences).shape[0] == 1 or np.unique(differences).shape[0] == 2) or seq_rest != 0# allow for 1 or 2**12-1
                # Perform calculations
                missed_packet_percent = round(n_missed_packets / total_number_of_packets * 100, 2)
                seq_loss_perc = round(seq_loss / total_pack_seq * 100, 2)
                duplicate_packet = n_duplicates 

                # Format times as strings
                time_string = f"{hours}:{minutes}:{seconds}"
                exp_time_string = f"{exp_hours}:{exp_minutes}:{exp_seconds}"

                mean_break_time = round(np.mean(avg_break_time), 5)
                std_break_time = round(np.std(avg_break_time), 5)
                mean_sampling_period = round(np.mean(sampling_period), 5)
                std_sampling_period = round(np.std(sampling_period), 5)
                max_sampling_period = round(np.max(sampling_period), 5)
                min_sampling_period = round(np.min(sampling_period), 5)

                if len(buffer_full_ts) > 0:
                    buffer_full_ts_mean = round(np.mean(buffer_full_ts), 5)
                    buffer_full_ts_std = round(np.std(buffer_full_ts), 5)
                    max_buffer_full_ts = round(np.max(buffer_full_ts), 5)
                    min_buffer_full_ts = round(np.min(buffer_full_ts), 5)
                else:
                    buffer_full_ts_mean, buffer_full_ts_std, max_buffer_full_ts, min_buffer_full_ts = 0,0,0,0
                if len(time_resets_ts) > 0:
                    max_time_resets_ts = round(np.max(time_resets_ts), 5)
                    min_time_resets_ts = round(np.min(time_resets_ts), 5)
                    mean_time_resets_ts = round(np.mean(time_resets_ts), 5)
                    std_time_resets_ts = round(np.std(time_resets_ts), 5)
                else:
                    max_time_resets_ts, min_time_resets_ts, mean_time_resets_ts, std_time_resets_ts = 0,0,0,0

                seq_loss_time = round(np.sum(seq_loss_ts), 5)
                if log:
                    if len(seq_loss_idx) > 0:
                        plt.figure()
                        plt.title(str(device.port) + " seq_loss_idx")
                        for index in seq_loss_idx:
                            plt.axvline(x=index, color='r', linestyle='--')  # Plots a vertical line at each index
                        #plt.plot(timestamps, ".")
                        plt.plot(timestamps, ".")
                        plt.ylabel("timestamps")
                        plt.show()
                        plt.close()

                # Append data to results_table
                results_table.append([
                    session_id, device.port, missed_packet_percent, n_missed_packets, total_number_of_packets, loss_points, total_loss_time, f"{mean_break_time} +- {std_break_time}",
                    duplicate_packet, time_string, exp_time_string, 
                    seq_rest,seq_loss, seq_loss_perc, seq_loss_time, sampling_rate,
                    mean_sampling_period, std_sampling_period, max_sampling_period, min_sampling_period,
                    buffer_full_event, buffer_lost_frames, np.sum(buffer_full_ts), f"{buffer_full_ts_mean} +- {buffer_full_ts_std}",
                    max_buffer_full_ts, min_buffer_full_ts, time_resets_count, max_time_resets_ts, min_time_resets_ts, mean_time_resets_ts, std_time_resets_ts
                ])
                print(f"SESSION: {session_id}; [Device = {device.port}] -> Missed packets = {n_missed_packets} (#) | Data Loss {missed_packet_percent} (%)")

            print("%%%%% SESSION END %%%%% \n")
        session.close()
        return results_table

    
db_path = "sqlite:///data/db.sqlite"  # This is the correct format for an SQLite URL  # Replace with your actual database path
data_analysis = DataAnalysis(db_url=db_path)

# Now, call the calc_data_loss method. You can pass the session_id and log flag if needed.
results_table = data_analysis.calc_data_loss(log=False) 

df = pd.DataFrame.from_dict(results_table) 
header = ['session_id', 'device', 'data loss (%)', 'lost frames (#)', 'total frames (#)', 'loss events (#)', 'total loss time (s)', 'avg loss time (s)', 'duplicates (#)', 'obt. duration (h:m:s)', \
          'exp. duration (h:m:s)', 'Seq Rest (#)', "Seq Loss (#)", "Seq Loss (%)", "Total Seq Loss (s)", 'exp sampling_rate (Hz)',\
            'avg sampling period (s)', 'std sampling period (s)', 'max sampling period (s)', 'min sampling period (s)', "buffer full (#)", "buffer lost frames (#)", "total time buffer full (s)",\
            "buffer full avg (ts)", "buffer full max (ts)", "buffer full min (ts)", "time resets (#)", "time resets max (ts)", "time resets min (ts)", "time resets avg (ts)", "time resets std (ts)"]
averages = df.mean(numeric_only=True)  # get the average of each numeric column
std_devs = df.std(numeric_only=True)  # get the standard deviation of each numeric column

summary_row = ['---' if col not in averages.index else f"{averages[col]:.2f} ± {std_devs[col]:.2f}" for col in df.columns]
df.loc['Average'] = summary_row
df.to_csv ('results/db_results.csv', index = True, header=header)

pdb.set_trace()

## QUESTIONS
# 1. In which case the timestamp is reset?
# 2. Why obtained duration > expected duration?
# 3. all the frames are being lost due to buffer full, why?
# 4. Parece que envia o dobro dos dados, envia o ts no tempo errado, mas mantem seq number
# 5. Buffer full is observed when timestamp goes back in time
# 6. led doesn't stop high frequency
# 7. timestamps are side by side why
#8. continue blinking after turn off

#TODO:
# 1. Increase timeout between saving, num works