"""
This class reads the data and returns dataframes and links
"""

import os
import pathlib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = pathlib.Path(__file__).parent.parent.parent.resolve()
Dataset_PATH = os.path.join(BASE_PATH, "Dataset/raw_data")
PARTICIPANT_PATH = os.path.join(Dataset_PATH, "participants.csv")

def read_participant_data() -> pd.DataFrame:
    """Read the participant data from the csv file

    Returns:
        pd.DataFrame: Dataframe containing the participant data
    """
    participant_df = pd.read_csv(PARTICIPANT_PATH)
    return participant_df

def create_participant_dict(path) -> dict[str, str]:
    csv_path = os.path.join(path, "csv")
    video_holo_path = os.path.join(csv_path, "video_data_holo.csv")
    eet_holo_path = os.path.join(csv_path, "eet_data_holo.csv")
    video_android_path = os.path.join(csv_path, "video_data_android.csv")
    sensor_android_path = os.path.join(csv_path, "sensor_data_android.csv")
    images_android = os.path.join(path, "images_android")
    images_holo = os.path.join(path, "images_holo")
    return {
        "base_path": str(path),
        "video_holo": video_holo_path,
        "eet_holo": eet_holo_path,
        "video_android": video_android_path,
        "sensor_android": sensor_android_path,
        "images_android": images_android,
        "images_holo": images_holo
    }

def get_dataset_paths(participants_list: list[str], long_data: bool, forward_data: bool, back_data: bool) -> dict:
    
    print(f"Processing participants: {participants_list}")
    res_dict = {}
    
    for participant in participants_list:
        participant_path = os.path.join(Dataset_PATH, participant)
        participant_path_obj = Path(participant_path)
        
        long_path = create_participant_dict(participant_path_obj / "long") if long_data and (participant_path_obj / "long").exists() else None
        forward_path = create_participant_dict(participant_path_obj / "forward") if forward_data and (participant_path_obj / "forward").exists() else None
        back_path = create_participant_dict(participant_path_obj / "back") if back_data and (participant_path_obj / "back").exists() else None
        
        res_dict[participant] = {
            "long": long_path,
            "forward": forward_path,
            "back": back_path
        }
    return res_dict