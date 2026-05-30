"""Axis-aligned bounding box used to define spatial evaluation regions."""


class EvaluationArea:
    """Rectangular area defined by two corner coordinates.

    Args:
        area_beginning: ``(x_min, y_min)`` corner of the bounding box.
        area_end: ``(x_max, y_max)`` corner of the bounding box.
    """

    def __init__(self, area_beginning: tuple, area_end: tuple) -> None:
        self.area_beginning = area_beginning
        self.area_end = area_end

    def is_within_area(self, coordinate: tuple) -> bool:
        """Return True if ``coordinate`` lies inside the bounding box."""
        x, y = coordinate
        return (
            self.area_beginning[0] <= x <= self.area_end[0] and
            self.area_beginning[1] <= y <= self.area_end[1]
        )
