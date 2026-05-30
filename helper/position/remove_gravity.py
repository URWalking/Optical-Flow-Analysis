"""Gravity removal from accelerometer data using an exponential moving average."""

import numpy as np
import pandas as pd


def remove_gravity_ema(
    df: pd.DataFrame,
    alpha: float = 0.95,
    cols: tuple[str, ...] = ("ax", "ay", "az"),
) -> pd.DataFrame:
    """Subtract a low-pass gravity estimate from raw accelerometer readings.

    An exponential moving average (EMA) with smoothing factor ``alpha`` is
    used to estimate the slowly-varying gravity component. The linear
    acceleration is then ``a_linear = a_raw - g_ema``.

    Args:
        df: DataFrame containing the raw accelerometer columns specified by
            ``cols``.
        alpha: EMA smoothing factor. Higher values make the gravity estimate
            respond more slowly (i.e. stronger low-pass filtering).
        cols: Names of the raw accelerometer columns (x, y, z order).

    Returns:
        Copy of ``df`` extended with ``ax_lin``, ``ay_lin``, ``az_lin``, and
        ``a_lin_mag`` columns.

    Raises:
        ValueError: If no finite accelerometer samples are found.
    """
    df = df.copy()
    a = df.loc[:, cols].to_numpy(dtype=float)

    g   = np.full_like(a, np.nan)
    lin = np.full_like(a, np.nan)

    valid = np.where(np.isfinite(a).all(axis=1))[0]
    if len(valid) == 0:
        raise ValueError("No finite accelerometer samples found.")

    i0 = valid[0]
    g[i0]   = a[i0]
    lin[i0] = a[i0] - g[i0]

    for i in range(i0 + 1, len(a)):
        if np.isfinite(a[i]).all() and np.isfinite(g[i - 1]).all():
            g[i]   = alpha * g[i - 1] + (1 - alpha) * a[i]
            lin[i] = a[i] - g[i]
        else:
            g[i]   = g[i - 1]
            lin[i] = np.nan

    df["ax_lin"]    = lin[:, 0]
    df["ay_lin"]    = lin[:, 1]
    df["az_lin"]    = lin[:, 2]
    df["a_lin_mag"] = np.linalg.norm(lin, axis=1)
    return df
