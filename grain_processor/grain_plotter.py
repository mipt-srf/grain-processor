import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import lognorm

from .grain_processor import GrainProcessor


class GrainPlotter:
    def __init__(self, processor: GrainProcessor) -> None:
        self.processor = processor

    def _lognorm_fit(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data = data[data > 0]
        shape, loc, scale = lognorm.fit(data, floc=0)
        x = np.linspace(0, np.quantile(data, 0.99), 100)
        pdf = lognorm.pdf(x, shape, loc, scale)
        return x, pdf

    def _plot_distribution(
        self, data: np.ndarray, xlabel: str, probability: bool = True, fit: bool = True
    ) -> None:
        max_data = np.quantile(data, 0.99)
        bins = np.linspace(0, max_data, 50)
        plt.hist(data, bins=bins, color="teal", alpha=0.6)
        plt.xlabel(xlabel)
        plt.ylabel("Count")

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

        plt.show()

    def plot_area_fractions(
        self, bins: int = 50, return_fig: bool = False
    ) -> Figure | None:
        fig, ax = plt.subplots()
        diameters = self.processor.get_diameters(in_nm=True)
        areas = self.processor.get_areas(in_nm=True)

        # Remove outliers
        max_diameter = np.quantile(diameters, 0.99)
        mask = diameters <= max_diameter
        diameters = diameters[mask]
        areas = areas[mask]

        # Define bin edges
        bin_edges = np.linspace(diameters.min(), diameters.max(), bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Sum areas for each bin
        area_sums, _ = np.histogram(diameters, bins=bin_edges, weights=areas)
        percentage = (area_sums / area_sums.sum()) * 100

        ax.bar(
            bin_centers,
            percentage,
            width=bin_edges[1] - bin_edges[0],
            color="teal",
            alpha=0.6,
        )
        ax.set_xlabel("Grain diameter, nm")
        ax.set_ylabel("Area fraction, %")

        if return_fig:
            return fig
        return None

    def plot_area_fractions_vs_perimeter(
        self, bins: int = 50, return_fig: bool = False
    ) -> Figure | None:
        fig, ax = plt.subplots()
        perimeters = self.processor.get_perimeters(in_nm=True)
        areas = self.processor.get_areas(in_nm=True)

        # Remove outliers
        max_diameter = np.quantile(perimeters, 0.99)
        mask = perimeters <= max_diameter
        perimeters = perimeters[mask]
        areas = areas[mask]

        # Define bin edges
        bin_edges = np.linspace(perimeters.min(), perimeters.max(), bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Sum areas for each bin
        area_sums, _ = np.histogram(perimeters, bins=bin_edges, weights=areas)
        percentage = (area_sums / area_sums.sum()) * 100

        ax.bar(
            bin_centers,
            percentage,
            width=bin_edges[1] - bin_edges[0],
            color="teal",
            alpha=0.6,
        )
        ax.set_xlabel("Grain diameter, nm")
        ax.set_ylabel("Area fraction, %")

        if return_fig:
            return fig
        return None

    def plot_diameters(
        self, fit: bool = True, probability: bool = True, return_fig: bool = False
    ) -> Figure | None:
        fig = plt.figure()
        diameters = self.processor.get_diameters(in_nm=True)

        label = "Grain diameter, "
        if self.processor.nm_per_pixel is not None:
            label += "nm"
        else:
            diameters = self.processor.get_diameters(in_nm=False)
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
        perimeters = self.processor.get_perimeters(in_nm=True)

        label = r"Grain perimeter, "
        if self.processor.nm_per_pixel is not None:
            label += "nm"
        else:
            perimeters = self.processor.get_perimeters(in_nm=False)
            label += "px"

        self._plot_distribution(
            data=perimeters, xlabel=label, probability=probability, fit=fit
        )
        if return_fig:
            return fig
        return None

    def plot_areas(
        self, fit: bool = False, probability: bool = True, return_fig: bool = False
    ) -> Figure | None:
        fig = plt.figure()

        label = r"Grain area, "
        if self.processor.nm_per_pixel is not None:
            areas = self.processor.get_areas(in_nm=True)
            label += "nm$^2$"
        else:
            areas = self.processor.get_areas(in_nm=False)
            label += "px$^2$"

        self._plot_distribution(
            data=areas, xlabel=label, probability=probability, fit=fit
        )
        if return_fig:
            return fig
        return None

    def plot_perimeters_vs_diameters(
        self, return_fig: bool = False, fit: bool = False
    ) -> Figure | None:
        fig = plt.figure()
        diameters = self.processor.get_diameters(in_nm=True)
        perimeters = self.processor.get_perimeters(in_nm=True)

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
