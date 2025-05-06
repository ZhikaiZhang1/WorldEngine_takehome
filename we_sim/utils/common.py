"""
Common utility functions and classes.
"""


def quartic(t: float) -> float:
    """
    Quartic function for smooth transitions.

    Args:
        t: Input value between -1 and 1.

    Returns:
        Quartic function value.
    """
    return 0 if abs(t) > 1 else (1 - t**2) ** 2


def blend_coef(t: float, duration: float, std: float) -> float:
    """
    Calculate blend coefficient for smooth transitions.

    Args:
        t: Current time.
        duration: Total duration.
        std: Standard deviation for the quartic function.

    Returns:
        Blend coefficient.
    """
    normalised_time = 2 * t / duration - 1
    return quartic(normalised_time / std)


def unit_smooth(normalised_time: float) -> float:
    """
    Smooth unit function.

    Args:
        normalised_time: Normalized time between 0 and 1.

    Returns:
        Smoothed value.
    """
    return 3 * normalised_time**2 - 2 * normalised_time**3
