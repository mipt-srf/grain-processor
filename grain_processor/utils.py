"""
Utility functions for grain image processing.
"""

from functools import wraps
from typing import Any, Callable

import numpy as np
from cv2.typing import MatLike
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def plot_decorator(func: Callable[..., MatLike]) -> Callable[..., MatLike]:
    """
    Decorator to optionally plot the image returned by a function.

    :param func: The function returning an image.
    :return: The wrapped function that displays the image if 'plot' is True.
    """

    @wraps(func)
    def wrapper(*args: Any, plot: bool = False, **kwargs: Any) -> MatLike:
        # call the original function to get the result
        result = func(*args, **kwargs)

        # if plot=True, plot the result
        if plot:
            plt.figure()
            plt.imshow(result, cmap="gray")
            plt.title(func.__name__.replace("_", " ").lstrip().capitalize())
            plt.axis("off")

        return result

    return wrapper


def get_hist_data(
    data: NDArray[np.float64], nm_per_bin: float, quantile: float = 0.995, weights: NDArray[np.float64] | None = None
) -> tuple[NDArray[np.float64], NDArray]:
    """
    Compute histogram data for a given array of values.

    :param data: The array of measurement data.
    :param nm_per_bin: The width of each histogram bin.
    :param quantile: The quantile used to determine the histogram range.
    :param weights: Optional weights for the histogram computation.
    :return: A tuple with bin edges (excluding the last edge) and histogram counts.
    """
    max_data = np.quantile(data, quantile)
    bins = np.arange(0.5, max_data + 1, nm_per_bin, dtype=np.float64)
    hist, bins = np.histogram(data, bins=bins, weights=weights)
    return bins[:-1], hist
