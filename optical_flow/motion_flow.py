import cv2
import numpy as np
from typing import Literal


def flow_lucas_kanade(path_prev: str, path_curr: str) -> float:
    """Compute horizontal optical flow between two frames using Lucas-Kanade.

    Tracks sparse feature points (good corners) from the previous frame into
    the current frame and returns the median x-displacement. Using the median
    makes the result robust against independently moving objects (e.g. people).

    Returns 0.0 if either image cannot be loaded or no features are found.
    """
    img_prev = cv2.imread(path_prev)
    img_curr = cv2.imread(path_curr)

    if img_prev is None or img_curr is None:
        return 0.0

    prev_gray = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)

    feature_params = dict(
        maxCorners=50,
        qualityLevel=0.1,
        minDistance=7,
        blockSize=7,
    )
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    if p0 is None:
        return 0.0

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        deltas_x = good_new[:, 0] - good_old[:, 0]
        if len(deltas_x) > 0:
            return float(np.median(deltas_x))

    return 0.0


def flow_farneback(path_prev: str, path_curr: str, resize_width: int = 250) -> float:
    """Compute horizontal optical flow between two frames using Farneback dense flow.

    Resizes both frames for performance, computes dense flow, and returns the
    median horizontal displacement (dx). The median is more robust than the
    mean when parts of the scene contain independently moving objects.

    Returns 0.0 if either image cannot be loaded.
    """
    img_prev = cv2.imread(path_prev)
    img_curr = cv2.imread(path_curr)

    if img_prev is None or img_curr is None:
        return 0.0

    prev_gray = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)

    h, w = prev_gray.shape
    new_h = int(h * (resize_width / w))
    prev_small = cv2.resize(prev_gray, (resize_width, new_h))
    curr_small = cv2.resize(curr_gray, (resize_width, new_h))

    flow_field = cv2.calcOpticalFlowFarneback(
        prev_small, curr_small, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )

    return float(np.median(flow_field[..., 0]))


def flow(
    path_prev: str,
    path_curr: str,
    resize_width: int = 250,
    algrthm: Literal["farneback", "lucas-kanade"] = "farneback",
) -> float:
    """Dispatch to the selected optical flow algorithm."""
    if algrthm == "farneback":
        return flow_farneback(path_prev, path_curr, resize_width)
    return flow_lucas_kanade(path_prev, path_curr)
