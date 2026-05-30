"""Parse raw accelerometer strings from the prototype data format.

The prototype records sensor data as stringified Python literals, e.g.
``"(123456789, [0.1, -0.2, 9.8])"``. This module parses that format into
individual numeric columns.
"""

import ast

import numpy as np
import pandas as pd


def parse_acc(cell) -> pd.Series:
    """Parse a single accelerometer cell into (t_acc, ax, ay, az).

    Args:
        cell: A string of the form ``"(timestamp, [ax, ay, az, ...])"`` or an
            empty / non-string value.

    Returns:
        A :class:`pd.Series` with index ``["t_acc", "ax", "ay", "az"]``.
        All values are ``NaN`` if parsing fails.
    """
    nan_row = pd.Series([np.nan, np.nan, np.nan, np.nan], index=["t_acc", "ax", "ay", "az"])

    if not isinstance(cell, str) or cell.strip() == "":
        return nan_row
    try:
        t, vec = ast.literal_eval(cell)
        ax, ay, az = vec[:3]
        return pd.Series([t, ax, ay, az], index=["t_acc", "ax", "ay", "az"])
    except Exception:
        return nan_row
