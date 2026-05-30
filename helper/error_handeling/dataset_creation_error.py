"""Validation helpers for dataset construction."""


def create_mismatch_error(len_list: int = 0, len_participants: int = 0, label: str = "") -> None:
    """Raise if the number of loaded trajectories exceeds the participant count.

    Args:
        len_list: Number of trajectories that were loaded.
        len_participants: Expected upper bound (number of participants).
        label: Human-readable label used in the error message (e.g. "Back").

    Raises:
        ValueError: If ``len_list > len_participants``.
    """
    if len_list > len_participants:
        raise ValueError(
            f"CRITICAL: {label} participant mismatch — {len_list} trajectories for {len_participants} participants."
        )
