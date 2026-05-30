"""Synchronise HoloLens, Android video, and Android sensor streams by timestamp."""

import pandas as pd


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename merged columns to a clean, unified schema."""
    df.rename(columns={
        "timestamp_unix":      "timestamp",
        "image_filename":      "holo_image_filename",
        "filename_pixel":      "android_image_filename",
        "ax_pixel_sensor":     "ax_sensor",
        "ay_pixel_sensor":     "ay_sensor",
        "az_pixel_sensor":     "az_sensor",
        "gx_pixel_sensor":     "gx_sensor",
        "gy_pixel_sensor":     "gy_sensor",
        "gz_pixel_sensor":     "gz_sensor",
    }, inplace=True)

    return df[[
        "timestamp",
        "holo_image_filename", "android_image_filename",
        "ax_sensor", "ay_sensor", "az_sensor",
        "gx_sensor", "gy_sensor", "gz_sensor",
        "pose",
    ]]


def synchronise(data_obj: dict, analysis: bool = False):
    """Align HoloLens video, Android video, and Android sensor data by timestamp.

    All three streams are trimmed to their common time window and merged with
    ``merge_asof`` (nearest-neighbour, 17 ms tolerance ≈ one frame at 60 Hz).
    The HoloLens timestamp is used as the master reference.

    Args:
        data_obj: Dict with keys ``"video_holo"``, ``"video_android"``, and
            ``"sensor_android"``, each holding a DataFrame.
        analysis: If True, also return a dict with jitter statistics.

    Returns:
        Merged DataFrame (and optionally a jitter stats dict when
        ``analysis=True``).
    """
    df_holo   = data_obj["video_holo"]
    df_android = data_obj["video_android"]
    df_sensor  = data_obj["sensor_android"]

    # Trim to common time window
    common_start = max(df_holo["timestamp_unix"].iloc[0],  df_android["timestamp_unix"].iloc[0])
    common_end   = min(df_holo["timestamp_unix"].iloc[-1], df_android["timestamp_unix"].iloc[-1])

    df_holo   = df_holo[(df_holo["timestamp_unix"] >= common_start) & (df_holo["timestamp_unix"] <= common_end)].copy()
    df_android = df_android[(df_android["timestamp_unix"] >= common_start) & (df_android["timestamp_unix"] <= common_end)].copy()
    df_sensor  = df_sensor[(df_sensor["timestamp_unix"] >= common_start) & (df_sensor["timestamp_unix"] <= common_end)].copy()

    df_holo    = df_holo.reset_index(drop=True)
    df_android = df_android.reset_index(drop=True)
    df_sensor  = df_sensor.reset_index(drop=True)

    df_android = df_android.add_suffix("_pixel")
    df_sensor  = df_sensor.add_suffix("_pixel_sensor")

    df_holo    = df_holo.sort_values("timestamp_unix")
    df_android = df_android.sort_values("timestamp_unix_pixel")
    df_sensor  = df_sensor.sort_values("timestamp_unix_pixel_sensor")

    # Merge 1: HoloLens + Android video (~17 ms tolerance for 60 Hz)
    df_merged = pd.merge_asof(
        df_holo, df_android,
        left_on="timestamp_unix",
        right_on="timestamp_unix_pixel",
        direction="nearest",
        tolerance=0.017,
    )

    # Merge 2: result + Android sensor
    df_merged = pd.merge_asof(
        df_merged, df_sensor,
        left_on="timestamp_unix",
        right_on="timestamp_unix_pixel_sensor",
        direction="nearest",
        tolerance=0.017,
    )

    # Sync quality report
    total        = len(df_merged)
    n_video      = df_merged["timestamp_unix_pixel"].notna().sum()
    n_sensor     = df_merged["timestamp_unix_pixel_sensor"].notna().sum()
    valid_sensor = df_merged.dropna(subset=["timestamp_unix_pixel_sensor"]).copy()

    print("--- Synchronisation Result ---")
    print(f"Total rows (HoloLens base): {total}")
    print(f"Video matches:  {n_video} ({n_video / total * 100:.2f}%)")
    print(f"Sensor matches: {n_sensor} ({n_sensor / total * 100:.2f}%)")

    diff = (valid_sensor["timestamp_unix"] - valid_sensor["timestamp_unix_pixel_sensor"]).abs() * 1000
    if not valid_sensor.empty:
        print(f"Sensor avg time offset: {diff.mean():.2f} ms")
        print(f"Sensor max time offset: {diff.max():.2f} ms")
    else:
        print("Warning: no sensor matches found.")

    if analysis:
        return df_merged, {"avg_jitter": diff.mean(), "max_jitter": diff.max()}

    return rename_columns(df_merged)
