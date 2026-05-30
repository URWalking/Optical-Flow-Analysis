"""Convert global position data to an ego-centric perspective.

The ground-truth trajectories are recorded in global coordinates. This module
converts them to an ego-centric frame so that forward movement is always
aligned with the positive x-axis, regardless of the walking direction.
"""

import numpy as np
import pandas as pd


def calc_ego_perspective_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ego-centric deltas from global grid-aligned positions.

    For each step the dominant axis (horizontal vs. vertical) is determined.
    Movement along that axis is mapped to the ego forward direction (ego_dx),
    while lateral deviation is mapped to ego_dy.

    Args:
        df: DataFrame with ``x_pos`` and ``y_pos`` columns.

    Returns:
        DataFrame extended with ``ego_dx`` and ``ego_dy`` columns.
    """
    df["dx1"] = df["x_pos"].shift(-1) - df["x_pos"]
    df["dy1"] = df["y_pos"].shift(-1) - df["y_pos"]
    df = df.dropna(subset=["dx1", "dy1"]).copy()

    dx = df["dx1"].to_numpy()
    dy = df["dy1"].to_numpy()
    horizontal = np.abs(dx) >= np.abs(dy)

    ego_x = np.zeros_like(dx)
    ego_y = np.zeros_like(dy)

    # Moving east
    m = horizontal & (dx >= 0);  ego_x[m] =  dx[m]; ego_y[m] =  dy[m]
    # Moving west
    m = horizontal & (dx < 0);   ego_x[m] = -dx[m]; ego_y[m] = -dy[m]
    # Moving north
    m = (~horizontal) & (dy >= 0); ego_x[m] =  dy[m]; ego_y[m] = -dx[m]
    # Moving south
    m = (~horizontal) & (dy < 0);  ego_x[m] = -dy[m]; ego_y[m] =  dx[m]

    df["ego_dx"] = ego_x
    df["ego_dy"] = ego_y
    return df
