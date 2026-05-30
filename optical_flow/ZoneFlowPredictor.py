"""Predict turning direction (left / straight / right) from optical flow within a zone."""

import numpy as np
import pandas as pd
from typing import Literal

from optical_flow.motion_flow import flow


Direction = Literal["links", "gerade", "rechts"]


class Predictor:
    """Classify whether a person turns left, goes straight, or turns right.

    The prediction is based on the smoothed horizontal optical flow signal
    computed from egocentric camera frames while the person is inside a
    defined zone radius around a decision point.

    Args:
        zone_radius: Spatial radius (same unit as x_new/y_new) that defines
            the decision zone around a corner coordinate.
        threshold: Minimum absolute mean flow required to classify as a turn.
            Values below this are classified as straight.
        global_bias: Constant bias subtracted from the smoothed flow signal
            to correct for systematic camera drift.
        algrthm: Optical flow algorithm to use.
    """

    def __init__(
        self,
        zone_radius: int = 500,
        threshold: float = 1.0,
        global_bias: float = 0.0,
        algrthm: Literal["farneback", "lucas-kanade"] = "lucas-kanade",
    ) -> None:
        self.zone_radius = zone_radius
        self.threshold = threshold
        self.global_bias = global_bias
        self.algrthm = algrthm

    def motion_calculation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-frame optical flow and add smoothed signal to the DataFrame.

        Adds two columns:
            - ``optical_flow_dx``: raw frame-to-frame horizontal displacement.
            - ``optical_flow_smooth``: rolling mean of dx with bias correction.
        """
        image_paths = df["android_image_filename"].tolist()
        flow_values = [0.0]

        for i in range(1, len(image_paths)):
            try:
                val = flow(image_paths[i - 1], image_paths[i], resize_width=250, algrthm=self.algrthm)
            except Exception:
                val = 0.0
            flow_values.append(val)

        df["optical_flow_dx"] = flow_values
        df["optical_flow_smooth"] = (
            df["optical_flow_dx"]
            .rolling(window=45, center=True)
            .mean()
            .fillna(0)
        )
        df["optical_flow_smooth"] -= self.global_bias
        return df

    def get_prediction_focus(self, series_data: pd.Series) -> Direction:
        """Classify a flow signal segment as left, straight, or right."""
        mean_val = series_data.mean()

        if abs(mean_val) < self.threshold:
            return "gerade"
        return "links" if mean_val > 0 else "rechts"

    def moved(self, df: pd.DataFrame, zone_pos: tuple = (0, 0)) -> Direction:
        """Predict turning direction for the frames inside the decision zone.

        Args:
            df: Trajectory DataFrame containing ``x_new``, ``y_new``, and
                ``android_image_filename`` columns.
            zone_pos: (cx, cy) centre of the decision zone.

        Returns:
            ``"links"``, ``"gerade"``, or ``"rechts"``.
        """
        cx, cy = zone_pos
        dist = np.sqrt((df["x_new"] - cx) ** 2 + (df["y_new"] - cy) ** 2)
        df_zone = df.loc[dist <= self.zone_radius].copy()
        df_motion = self.motion_calculation(df_zone)
        return self.get_prediction_focus(df_motion["optical_flow_smooth"])
