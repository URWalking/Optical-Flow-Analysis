"""Coordinate alignment utilities: jump removal, PCA rotation, and Manhattan rectification."""

import ast

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _fine_tune_rotation(coords: np.ndarray) -> np.ndarray:
    """Apply a small rotation so dominant movement directions align with the grid axes."""
    if len(coords) < 2:
        return coords

    diffs = np.diff(coords, axis=0)
    complex_diffs = diffs[:, 0] + 1j * diffs[:, 1]
    lengths = np.abs(complex_diffs)
    mask = lengths > 0.5

    if np.sum(mask) < 5:
        return coords

    mean_orientation = np.sum(complex_diffs[mask] ** 4)
    grid_angle = np.angle(mean_orientation) / 4.0

    c, s = np.cos(-grid_angle), np.sin(-grid_angle)
    R = np.array([[c, -s], [s, c]])
    rotated = coords @ R.T
    rotated -= rotated[0]
    return rotated


def transform_coordinates_test_area(
    df: pd.DataFrame,
    jump_threshold: float = 5.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract, clean, and PCA-align XZ position coordinates from a trajectory DataFrame.

    Steps:
        1. Parse the ``pose`` column and extract the XZ plane (columns 0 and 2
           of the fourth position element).
        2. Remove jump artefacts where consecutive distance exceeds
           ``jump_threshold``.
        3. Rotate using PCA so the primary movement axis aligns with X.
        4. Ensure the trajectory starts at the origin and moves in the
           positive direction.
        5. Apply a fine-tuning rotation based on the upper-percentile slope.

    Returns:
        A tuple of ``(coords, indices)`` where ``coords`` is an (N, 2) array
        in metres and ``indices`` are the corresponding DataFrame row labels.
        Returns ``(None, None)`` if there are insufficient valid samples.
    """
    valid_indices = []
    raw_pose = []

    for idx, val in df["pose"].items():
        if isinstance(val, str):
            try:
                p = np.array(ast.literal_eval(val))
                if len(p) >= 4:
                    raw_pose.append(p)
                    valid_indices.append(idx)
            except Exception:
                continue

    if len(raw_pose) < 30:
        return None, None

    coords = np.array([p[3] for p in raw_pose])[:, [0, 2]]
    current_indices = np.array(valid_indices)

    # Remove jump artefacts
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    mask = np.ones(len(coords), dtype=bool)
    for idx in np.where(diffs > jump_threshold)[0]:
        mask[idx + 1] = False

    coords = coords[mask]
    current_indices = current_indices[mask]

    if len(coords) < 2:
        return None, None

    # PCA rotation
    pca = PCA(n_components=2)
    pca.fit(coords - np.mean(coords, axis=0))
    rotated = coords @ pca.components_.T
    rotated -= rotated[0]

    # Ensure primary axis is the longer one
    width  = np.max(rotated[:, 0]) - np.min(rotated[:, 0])
    height = np.max(rotated[:, 1]) - np.min(rotated[:, 1])
    if height > width:
        rotated = rotated[:, [1, 0]]
        rotated -= rotated[0]

    # Polarity correction
    if np.mean(rotated[:, 0]) < 0:
        rotated[:, 0] *= -1
    if np.mean(rotated[:, 1]) < 0:
        rotated[:, 1] *= -1
    rotated -= rotated[0]

    # Fine-tune rotation using the upper-percentile slope
    upper_threshold = np.percentile(rotated[:, 1], 80)
    upper_points = rotated[rotated[:, 1] > upper_threshold]

    if len(upper_points) > 5:
        slope, _ = np.polyfit(upper_points[:, 0], upper_points[:, 1], 1)
        angle = np.arctan(slope)
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]])
        rotated = rotated @ R.T
        rotated -= rotated[0]

    return rotated, current_indices


def transform_coordinates(
    df: pd.DataFrame,
    jump_threshold: float = 5.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Full coordinate pipeline: PCA alignment → grid correction → Manhattan rectification.

    Builds on :func:`transform_coordinates_test_area` and adds:
        - Dominant-angle correction so movement aligns with the 90° grid.
        - Signed-area polarity check to enforce consistent orientation.
        - Manhattan rectification: each step is forced to be either purely
          horizontal or purely vertical.

    Returns:
        ``(coords, indices)`` or ``(None, None)`` on failure.
    """
    coords, indices = transform_coordinates_test_area(df, jump_threshold)

    if coords is None or len(coords) < 10:
        return None, None

    # Dominant-angle correction (quantise to nearest 90° grid line)
    diffs = np.diff(coords, axis=0)
    angles_mod = np.arctan2(diffs[:, 1], diffs[:, 0]) % (np.pi / 2)
    hist, bin_edges = np.histogram(angles_mod, bins=90)
    dominant_angle = bin_edges[np.argmax(hist)]

    c, s = np.cos(-dominant_angle), np.sin(-dominant_angle)
    R = np.array([[c, -s], [s, c]])
    coords = coords @ R.T

    # Mirror and orientation consistency
    coords[:, 1] *= -1
    x, y = coords[:, 0], coords[:, 1]
    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if area < 0:
        coords[:, 1] *= -1
    if np.median(coords[:, 0]) < 0:
        coords[:, 0] *= -1

    # Manhattan rectification
    rectified = coords.copy()
    for i in range(1, len(rectified)):
        dx = abs(rectified[i, 0] - rectified[i - 1, 0])
        dy = abs(rectified[i, 1] - rectified[i - 1, 1])
        if dx > dy:
            rectified[i, 1] = rectified[i - 1, 1]
        else:
            rectified[i, 0] = rectified[i - 1, 0]

    rectified -= rectified[0]
    rectified[:, 1] *= -1

    return rectified, indices
