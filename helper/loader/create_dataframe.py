"""Build synchronised, coordinate-aligned DataFrames from raw multi-modal recordings.

Images are automatically resized to TARGET_SIZE on first use and cached in
sibling ``*_small`` folders to avoid repeated I/O overhead.
"""

import multiprocessing
import os

import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from tqdm import tqdm

from helper.error_handeling.dataset_creation_error import create_mismatch_error
from helper.loader.evaulation_area import EvaluationArea
from helper.loader.position_alignment import transform_coordinates, transform_coordinates_test_area
from helper.loader.process_helper import get_specific_data
from helper.sync.synchronise import synchronise

TARGET_SIZE = (256, 256)

FOLDER_MAPPINGS = [
    ("images_holo",    "images_holo_small",    "image_filename"),
    ("images_android", "images_android_small", "filename"),
]


# ---------------------------------------------------------------------------
# Image resizing (runs in a separate process via ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def resize_single_image_worker(args: tuple[str, str]) -> None:
    """Resize one image from ``src`` to ``dst`` at TARGET_SIZE.

    Skips silently if the destination already exists or if the source is
    unreadable. Thread-safe: the destination directory is created with
    ``exist_ok=True``.
    """
    src, dst = args
    if os.path.exists(dst):
        return
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with Image.open(src) as img:
            img_small = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            img_rotated = img_small.rotate(-90, expand=True)
            img_rotated.save(dst, quality=95)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def get_data(
    dict_paths: dict,
    long_data: bool,
    back_data: bool,
    forward_data: bool,
    android_included: bool,
    holo_included: bool,
    participant_ls: list,
) -> dict:
    list_csv = ["video_holo"]
    if android_included:
        list_csv.extend(["video_android", "sensor_android"])
    return get_specific_data(
        data=dict_paths,
        long_data=long_data,
        back_data=back_data,
        forward_data=forward_data,
        android=android_included,
        holo=holo_included,
        list_csv=list_csv,
        participant_ls=participant_ls,
    )


def get_data_package(data_obj: dict) -> dict:
    """Extract the observable modalities from a participant/mode sub-dict."""
    return {
        item: data_obj[item]
        for item in ("video_holo", "video_android", "sensor_android")
        if item in data_obj
    }


def filter_areas(
    df: pd.DataFrame,
    eval_area: EvaluationArea = None,
    train: bool = False,
    all_data: bool = False,
) -> pd.DataFrame:
    """Spatially filter a trajectory DataFrame.

    Args:
        df: Trajectory with ``x_new`` / ``y_new`` columns.
        eval_area: If given, restrict to (or exclude from) the bounding box.
        train: If True, return data *outside* the eval area (training split).
        all_data: If True, return the full trajectory ignoring the eval area.
    """
    if eval_area is not None:
        x_min, y_min = eval_area.area_beginning
        x_max, y_max = eval_area.area_end
        x_min, y_min, x_max, y_max = x_min * 100, y_min * 100, x_max * 100, y_max * 100
        mask = (
            (df["x_new"] >= x_min) & (df["x_new"] <= x_max) &
            (df["y_new"] >= y_min) & (df["y_new"] <= y_max)
        )
        df_clean = df[~mask].copy() if train else df[mask].copy()
    else:
        df_clean = df.copy()

    if not df_clean.empty and not train:
        df_clean["x_new"] -= df_clean["x_new"].iloc[0]
        df_clean["y_new"] -= df_clean["y_new"].iloc[0]

    if all_data:
        df_clean = df.copy()

    if train or all_data:
        origin_mask = (
            (df_clean["x_new"] >= -5) & (df_clean["x_new"] <= 5) &
            (df_clean["y_new"] >= -5) & (df_clean["y_new"] <= 5)
        )
        df_clean = df_clean[~origin_mask].copy()

    return df_clean


def prepare_data(data_dict: dict, key_way: str = "forward") -> list[pd.DataFrame]:
    """Synchronise and align all trajectories for one walk direction.

    Args:
        data_dict: Nested dict keyed by participant → mode → modality.
        key_way: One of ``"forward"``, ``"back"``, or ``"long"``.

    Returns:
        List of cleaned DataFrames with ``x_new`` / ``y_new`` columns in cm.
    """
    packages = [
        get_data_package(data_dict[key][key_way])
        for key in data_dict
        if key_way in data_dict[key]
    ]
    synced = [synchronise(pkg, analysis=False) for pkg in packages]

    result = []
    for df in synced:
        if key_way == "long":
            coords, indices = transform_coordinates(df)
        else:
            coords, indices = transform_coordinates_test_area(df)

        if coords is None:
            continue

        df_clean = df.loc[indices].copy()
        df_clean["x_new"] = coords[:, 0] * 100
        df_clean["y_new"] = coords[:, 1] * 100
        df_clean.dropna(inplace=True)
        result.append(df_clean)

    return result


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class CreateDataset:
    """High-level interface for loading, resizing, synchronising, and filtering data.

    Args:
        dict_paths: Nested path dict as returned by ``get_dataset_paths``.
        eval_areas: Dict mapping walk direction to an :class:`EvaluationArea`
            (or ``None`` to use the whole trajectory).
        train_participants: Participant IDs used for training splits.
        eval_participants: Participant IDs used for evaluation splits.
        android_included: Whether to load Android video / sensor streams.
        holo_included: Whether to load HoloLens video / EET streams.
    """

    def __init__(
        self,
        dict_paths: dict,
        eval_areas: dict = None,
        train_participants: list = None,
        eval_participants: list = None,
        android_included: bool = False,
        holo_included: bool = False,
    ) -> None:
        self.dict_paths = dict_paths
        self.eval_areas = eval_areas or {}
        self.train_participants = train_participants or []
        self.eval_participants = eval_participants or []
        self.android_included = android_included
        self.holo_included = holo_included

    # ------------------------------------------------------------------
    # Image resizing
    # ------------------------------------------------------------------

    def _ensure_resized_images(self, data_dict: dict) -> dict:
        """Create ``*_small`` image folders in parallel (skipped if they exist)."""
        tasks = []
        df_updates = []

        print("Checking/Creating resized images (256x256)...")

        for participant, modes in data_dict.items():
            for mode_key, datasets in modes.items():
                if datasets is None:
                    continue
                for ds_key, folder_signature in (
                    ("video_holo",    "images_holo"),
                    ("video_android", "images_android"),
                ):
                    if ds_key not in datasets or datasets[ds_key] is None:
                        continue

                    df = datasets[ds_key]
                    mapping = next((m for m in FOLDER_MAPPINGS if m[0] == folder_signature), None)
                    if mapping is None:
                        continue

                    old_folder, new_folder, col_name = mapping
                    if col_name not in df.columns:
                        continue

                    sample_path = str(df.iloc[0][col_name])
                    if old_folder not in sample_path:
                        continue

                    for p in df[col_name].tolist():
                        tasks.append((p, str(p).replace(old_folder, new_folder)))
                    df_updates.append((df, col_name, old_folder, new_folder))

        if tasks:
            if not os.path.exists(tasks[0][1]):
                print(f"Resizing {len(tasks)} images with multiprocessing...")
                workers = max(1, multiprocessing.cpu_count() - 2)
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    list(tqdm(executor.map(resize_single_image_worker, tasks), total=len(tasks)))
            else:
                print("Small images found. Skipping resize.")

        for df, col, old, new in df_updates:
            df[col] = df[col].astype(str).str.replace(old, new)

        return data_dict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_training_dataframe(
        self,
        long_data: bool = False,
        back_data: bool = False,
        forward_data: bool = False,
    ) -> dict[str, list[pd.DataFrame]]:
        """Return spatially filtered training trajectories per walk direction."""
        raw = get_data(
            dict_paths=self.dict_paths,
            long_data=long_data,
            back_data=back_data,
            forward_data=forward_data,
            android_included=self.android_included,
            holo_included=self.holo_included,
            participant_ls=self.train_participants,
        )
        self._ensure_resized_images(raw)

        ls_back = prepare_data(raw, "back")    if back_data    else []
        ls_fwd  = prepare_data(raw, "forward") if forward_data else []
        ls_long = prepare_data(raw, "long")    if long_data    else []

        print("Training material")
        create_mismatch_error(len(ls_back), len(self.train_participants), "Back")
        create_mismatch_error(len(ls_fwd),  len(self.train_participants), "Forward")
        create_mismatch_error(len(ls_long), len(self.train_participants), "Long")

        for key, eval_area in self.eval_areas.items():
            if eval_area is None:
                continue
            if key == "back" and back_data:
                ls_back = [filter_areas(df, eval_area, train=True) for df in ls_back]
            if key == "forward" and forward_data:
                ls_fwd  = [filter_areas(df, eval_area, train=True) for df in ls_fwd]
            if key == "long" and long_data:
                ls_long = [filter_areas(df, eval_area, train=True) for df in ls_long]

        return {"back": ls_back, "forward": ls_fwd, "long": ls_long}

    def get_eval_dataframes(
        self,
        long_data: bool = False,
        back_data: bool = False,
        forward_data: bool = True,
        whole_way_back: bool = False,
        whole_way_fwd: bool = False,
        whole_way_long: bool = False,
        test_area: bool = False,
        all_data: bool = True,
    ) -> dict[str, list[pd.DataFrame] | None]:
        """Return evaluation trajectories, optionally filtered to a test area."""
        raw = get_data(
            dict_paths=self.dict_paths,
            long_data=long_data,
            back_data=back_data,
            forward_data=forward_data,
            android_included=self.android_included,
            holo_included=self.holo_included,
            participant_ls=self.eval_participants,
        )
        self._ensure_resized_images(raw)

        ls_back = prepare_data(raw, "back")    if back_data    else []
        ls_fwd  = prepare_data(raw, "forward") if forward_data else []
        ls_long = prepare_data(raw, "long")    if long_data    else []

        create_mismatch_error(len(ls_back), len(self.eval_participants), "Back")
        create_mismatch_error(len(ls_fwd),  len(self.eval_participants), "Forward")
        create_mismatch_error(len(ls_long), len(self.eval_participants), "Long")

        test_area_back, test_area_forward, test_area_long = [], [], []

        if test_area:
            for key, eval_area in self.eval_areas.items():
                if eval_area is None:
                    continue
                if key == "back" and back_data:
                    test_area_back    = [filter_areas(df, eval_area, train=False) for df in ls_back]
                if key == "forward" and forward_data:
                    test_area_forward = [filter_areas(df, eval_area, train=False) for df in ls_fwd]
                if key == "long" and long_data:
                    test_area_long    = [filter_areas(df, eval_area, train=False) for df in ls_long]

        whole_back = [filter_areas(df, train=False, all_data=all_data) for df in ls_back]
        whole_fwd  = [filter_areas(df, train=False, all_data=all_data) for df in ls_fwd]
        whole_long = [filter_areas(df, train=False, all_data=all_data) for df in ls_long]

        return {
            "eval_back_way":          whole_back         if whole_way_back  else None,
            "eval_forward_way":       whole_fwd          if whole_way_fwd   else None,
            "eval_long_way":          whole_long         if whole_way_long  else None,
            "eval_test_back_way":     test_area_back     if test_area       else None,
            "eval_test_forward_way":  test_area_forward  if test_area       else None,
            "eval_test_long_way":     test_area_long     if test_area       else None,
        }
