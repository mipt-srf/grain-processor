import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import lognorm


class GrainPlotter:
    def __init__(
        self,
        diameters: np.ndarray,
        perimeters: np.ndarray,
        areas: np.ndarray,
        nm_per_pixel: float | None = None,
    ) -> None:
        self._diameters = diameters
        self._perimeters = perimeters
        self._areas = areas
        self._nm_per_pixel = nm_per_pixel

    def _lognorm_fit(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data = data[data > 0]
        shape, loc, scale = lognorm.fit(data, floc=0)
        x = np.linspace(0, np.quantile(data, 0.99), 100)
        pdf = lognorm.pdf(x, shape, loc, scale)
        return x, pdf

    def _plot_distribution(
        self,
        data: np.ndarray,
        xlabel: str,
        ylabel: str = "Count",
        nm_per_bin=1,
        probability: bool = True,
        fit: bool = True,
        weights=None,
        quantile=0.995,
    ) -> None:
        max_data = np.quantile(data, quantile)
        bins = np.arange(0.5, max_data + 1, nm_per_bin)
        plt.hist(data, bins=bins, color="teal", alpha=0.6, weights=weights)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if fit:
            x, pdf = self._lognorm_fit(data)
            bin_width = bins[1] - bins[0]
            pdf_scaled = pdf * len(data) * bin_width
            plt.plot(x, pdf_scaled, "r-", linewidth=2, color="lightcoral")

        if probability:
            plt.gca().yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: "{:.0f}".format(y / len(data) * 100))
            )
            plt.ylabel("Probability, %")

    def plot_area_fractions(
        self, nm_per_bin=1, return_fig: bool = False
    ) -> Figure | None:
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

    def plot_area_fractions_vs_perimeter(
        self, nm_per_bin=3.5, return_fig: bool = False
    ) -> Figure | None:
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

    def plot_diameters(
        self, fit: bool = True, probability: bool = True, return_fig: bool = False
    ) -> Figure | None:
        fig = plt.figure()
        diameters = self._diameters

        label = "Grain diameter, "
        if self._nm_per_pixel is not None:
            label += "nm"
        else:
            label += "px"

        self._plot_distribution(
            data=diameters, xlabel=label, probability=probability, fit=fit
        )
        if return_fig:
            return fig
        return None

    def plot_perimeters(
        self, fit: bool = True, probability: bool = True, return_fig: bool = False
    ) -> Figure | None:
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

    def plot_areas(
        self, fit: bool = False, probability: bool = True, return_fig: bool = False
    ) -> Figure | None:
        fig = plt.figure()

        label = r"Grain area, "
        if self._nm_per_pixel is not None:
            areas = self._areas
            label += "nm$^2$"
        else:
            areas = self._areas
            label += "px$^2$"

        self._plot_distribution(
            data=areas, xlabel=label, probability=probability, fit=fit, nm_per_bin=7.5
        )
        if return_fig:
            return fig
        return None

    def plot_perimeters_vs_diameters(
        self, return_fig: bool = False, fit: bool = False
    ) -> Figure | None:
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
