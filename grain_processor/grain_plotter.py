"""Module for plotting grain measurement distributions and relationships."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.stats import lognorm

from .utils import get_hist_data


class GrainPlotter:
    """
    A utility class to generate plots for grain diameters, perimeters, and areas.
    """

    def __init__(
        self,
        diameters: NDArray[np.float64],
        perimeters: NDArray[np.float64],
        areas: NDArray[np.float64],
        nm_per_pixel: float | None = None,
    ) -> None:
        """
        Initialize the GrainPlotter with grain measurements.

        :param diameters: Array of grain diameters.
        :param perimeters: Array of grain perimeters.
        :param areas: Array of grain areas.
        :param nm_per_pixel: Optional scale factor for unit conversion.
        """
        self._diameters = diameters
        self._perimeters = perimeters
        self._areas = areas
        self._nm_per_pixel = nm_per_pixel

    def _lognorm_fit(self, data: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Fit a log-normal distribution to the data and compute its PDF.

        :param data: Array of measurement data.
        :return: A tuple with evaluation points and corresponding PDF values.
        """
        data = data[data > 0]
        shape, loc, scale = lognorm.fit(data, floc=0)
        x = np.linspace(0, np.quantile(data, 0.99), 100, dtype=np.float64)
        pdf = lognorm.pdf(x, shape, loc, scale)
        return x, pdf

    def _plot_distribution(
        self,
        data: NDArray[np.float64],
        xlabel: str,
        ylabel: str = "Count",
        nm_per_bin: float = 1,
        probability: bool = True,
        fit: bool = True,
        weights: NDArray[np.float64] | None = None,
        quantile: float = 0.995,
    ) -> None:
        """
        Plot a histogram of the provided data and optionally overlay a log-normal fit.

        :param data: Array of measurement data.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        :param nm_per_bin: Bin width for the histogram.
        :param probability: Whether to display the y-axis as a percentage.
        :param fit: Whether to overlay a log-normal fit.
        :param weights: Optional weights for the histogram.
        :param quantile: Quantile to determine the histogram range.
        """
        bins, hist = get_hist_data(data=data, nm_per_bin=nm_per_bin, quantile=quantile, weights=weights)
        plt.bar(bins, hist, width=nm_per_bin, color="teal", alpha=0.6)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if fit:
            x, pdf = self._lognorm_fit(data)
            bin_width = nm_per_bin
            pdf_scaled = pdf * len(data) * bin_width
            plt.plot(x, pdf_scaled, "-", linewidth=2, color="lightcoral")

        if probability:
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0f}".format(y / len(data) * 100)))
            plt.ylabel("Probability, %")

    def plot_area_fractions(self, nm_per_bin: float = 1, return_fig: bool = False) -> Figure | None:
        """
        Plot the area fraction distribution based on grain diameters.

        :param nm_per_bin: Bin width for the histogram.
        :param return_fig: Whether to return the figure object.
        """
        fig = plt.figure()
        diameters = self._diameters
        areas = self._areas
        fractions = areas / areas.sum() * 100

        self._plot_distribution(
            data=diameters,
            xlabel="Grain diameter, nm",
            ylabel="Area fraction, %",
            nm_per_bin=nm_per_bin,
            probability=False,
            fit=False,
            weights=fractions,
        )

        if return_fig:
            return fig
        return None

    def plot_area_fractions_vs_perimeter(self, nm_per_bin: float = 3.5, return_fig: bool = False) -> Figure | None:
        """
        Plot the area fraction distribution with respect to grain perimeter.

        :param nm_per_bin: Bin width for the histogram.
        :param return_fig: Whether to return the figure object.
        """
        fig = plt.figure()
        perimeters = self._perimeters
        areas = self._areas
        fractions = areas / areas.sum() * 100

        self._plot_distribution(
            data=perimeters,
            xlabel="Grain perimeter, nm",
            ylabel="Area fraction, %",
            nm_per_bin=nm_per_bin,
            probability=False,
            fit=False,
            weights=fractions,
        )

        if return_fig:
            return fig
        return None

    def plot_diameters(self, fit: bool = True, probability: bool = True, return_fig: bool = False) -> Figure | None:
        """
        Plot the distribution of grain diameters.

        :param fit: Whether to overlay a log-normal fit.
        :param probability: Whether to display the histogram as a probability.
        :param return_fig: Whether to return the figure object.
        """
        fig = plt.figure()
        diameters = self._diameters

        label = "Grain diameter, "
        if self._nm_per_pixel is not None:
            label += "nm"
        else:
            label += "px"

        self._plot_distribution(data=diameters, xlabel=label, probability=probability, fit=fit)
        if return_fig:
            return fig
        return None

    def plot_perimeters(self, fit: bool = True, probability: bool = True, return_fig: bool = False) -> Figure | None:
        """
        Plot the distribution of grain perimeters.

        :param fit: Whether to overlay a log-normal fit.
        :param probability: Whether to display the histogram as a probability.
        :param return_fig: Whether to return the figure object.
        """
        fig = plt.figure()
        perimeters = self._perimeters

        label = r"Grain perimeter, "
        if self._nm_per_pixel is not None:
            label += "nm"
        else:
            label += "px"

        self._plot_distribution(
            data=perimeters,
            xlabel=label,
            probability=probability,
            fit=fit,
            nm_per_bin=3,
        )
        if return_fig:
            return fig
        return None

    def plot_areas(self, fit: bool = False, probability: bool = True, return_fig: bool = False) -> Figure | None:
        """
        Plot the distribution of grain areas.

        :param fit: Whether to overlay a log-normal fit.
        :param probability: Whether to display the histogram as a probability.
        :param return_fig: Whether to return the figure object.
        """
        fig = plt.figure()

        label = r"Grain area, "
        if self._nm_per_pixel is not None:
            areas = self._areas
            label += "nm$^2$"
        else:
            areas = self._areas
            label += "px$^2$"

        self._plot_distribution(data=areas, xlabel=label, probability=probability, fit=fit, nm_per_bin=7.5)
        if return_fig:
            return fig
        return None

    def plot_perimeters_vs_diameters(self, return_fig: bool = False, fit: bool = False) -> Figure | None:
        """
        Create a scatter plot to compare grain perimeters and diameters, with an optional linear fit.

        :param return_fig: Whether to return the figure object.
        :param fit: Whether to overlay a linear fit.
        """
        fig = plt.figure()
        diameters = self._diameters
        perimeters = self._perimeters

        plt.scatter(diameters, perimeters, color="teal", alpha=0.6)
        plt.xlabel("Grain diameter, nm")
        plt.ylabel("Grain perimeter, nm")

        if fit:
            coeffs = np.polyfit(diameters, perimeters, 1)
            fit_line = np.poly1d(coeffs)
            slope = coeffs[0]
            plt.plot(
                diameters,
                fit_line(diameters),
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.6,
            )
            plt.legend([f"Linear fit (slope={slope:.2f})", "Data"])

        plt.show()
        if return_fig:
            return fig
        return None
