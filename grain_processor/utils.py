from functools import wraps
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt


def plot_decorator(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, plot: bool = False, **kwargs) -> np.ndarray:
        # call the original function to get the result
        result = func(*args, **kwargs)

        # if plot=True, plot the result
        if plot:
            plt.figure()
            plt.imshow(result, cmap="gray")
            plt.title(func.__name__.replace("_", " ").lstrip().capitalize())
            plt.axis("off")
            plt.show()

        return result

    return wrapper
