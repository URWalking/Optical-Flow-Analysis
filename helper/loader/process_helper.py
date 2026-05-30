"""Low-level helpers for loading and filtering raw dataset files."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def get_dataframe(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Read a CSV file and return it as a DataFrame, or None if the path is invalid."""
    if not path:
        return None
    if not Path(path).exists():
        print(f"Path does not exist: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def add_absolute_paths(data: pd.DataFrame, base_path: str, col_name: str) -> pd.DataFrame:
    """Prepend ``base_path`` to every entry in ``col_name``."""
    data[col_name] = data[col_name].apply(lambda x: os.path.join(base_path, str(x)))
    return data


def get_specific_data(
    data: Dict[str, Any],
    long_data: bool = False,
    back_data: bool = False,
    forward_data: bool = False,
    android: bool = False,
    holo: bool = False,
    list_csv: list = None,
    participant_ls: list = None,
) -> Dict[str, Any]:
    """Load the requested modalities for the selected participants.

    Args:
        data: Nested path dict as returned by ``get_dataset_paths``.
        long_data: Include the longitudinal walk direction.
        back_data: Include the backward walk direction.
        forward_data: Include the forward walk direction.
        android: Load Android video and sensor streams.
        holo: Load HoloLens video and EET streams.
        list_csv: Explicit list of CSV keys to load (overrides android/holo flags
            for fine-grained control).
        participant_ls: If non-empty, restrict to these participant IDs.

    Returns:
        Nested dict ``{participant: {mode: {modality: DataFrame}}}``.
    """
    list_csv = list_csv or []
    participant_ls = participant_ls or []

    requested_modes = []
    if long_data:    requested_modes.append("long")
    if back_data:    requested_modes.append("back")
    if forward_data: requested_modes.append("forward")

    loaded_data: Dict[str, Any] = {}

    for participant, modes in data.items():
        if participant_ls and participant not in participant_ls:
            continue

        loaded_data[participant] = {}

        for mode in requested_modes:
            if mode not in modes or modes[mode] is None:
                continue

            mode_content = modes[mode]
            base_path = mode_content["base_path"]
            loaded_data[participant][mode] = {}

            if android:
                if "video_android" in mode_content and "video_android" in list_csv:
                    df = get_dataframe(mode_content["video_android"])
                    if df is not None:
                        df = add_absolute_paths(df, os.path.join(base_path, "images_android"), col_name="filename")
                    loaded_data[participant][mode]["video_android"] = df
                else:
                    loaded_data[participant][mode]["video_android"] = None

                if "sensor_android" in mode_content and "sensor_android" in list_csv:
                    loaded_data[participant][mode]["sensor_android"] = get_dataframe(mode_content["sensor_android"])
                else:
                    loaded_data[participant][mode]["sensor_android"] = None

            if holo:
                if "video_holo" in mode_content and "video_holo" in list_csv:
                    df = get_dataframe(mode_content["video_holo"])
                    if df is not None:
                        df = add_absolute_paths(df, os.path.join(base_path, "images_holo"), col_name="image_filename")
                    loaded_data[participant][mode]["video_holo"] = df
                else:
                    loaded_data[participant][mode]["video_holo"] = None

                if "eet_holo" in mode_content and "eet_holo" in list_csv:
                    loaded_data[participant][mode]["eet_holo"] = get_dataframe(mode_content["eet_holo"])
                else:
                    loaded_data[participant][mode]["eet_holo"] = None

    return loaded_data
