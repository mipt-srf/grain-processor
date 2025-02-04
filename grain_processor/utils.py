from functools import wraps
from typing import Callable

import numpy as np
from cv2.typing import MatLike
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def plot_decorator(func: Callable[..., MatLike]) -> Callable[..., MatLike]:
    @wraps(func)
    def wrapper(*args, plot: bool = False, **kwargs) -> MatLike:
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
    data: np.ndarray, nm_per_bin: float, quantile=0.995, weights=None
) -> tuple[NDArray[np.float64], NDArray]:
    max_data = np.quantile(data, quantile)
    bins = np.arange(0.5, max_data + 1, nm_per_bin, dtype=np.float64)
    hist, bins = np.histogram(data, bins=bins, weights=weights)
    return bins[:-1], hist
