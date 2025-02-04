"""Module providing functionality for processing and analyzing grain images using FFT filtering,
watershed segmentation, and statistical analysis.
"""

from pathlib import Path

import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from .grain_plotter import GrainPlotter
from .grain_segmenter import WatershedSegmenter
from .utils import get_hist_data, plot_decorator


class GrainProcessor:
    """
    Processes grain images by performing FFT filtering, segmentation, and analysis.

    This class reads an image, applies optional preprocessing, segments grains using the watershed algorithm,
    computes measurement statistics, and saves results.
    """

    def __init__(
        self,
        image_path: Path | str,
        cut_SEM: bool = False,
        fft_filter: bool = False,
        nm_per_pixel: float | None = None,
    ) -> None:
        """
        Initialize the GrainProcessor.

        :param image_path: Path to the input image.
        :param cut_SEM: Whether to perform image cutting (for SEM images).
        :param fft_filter: Whether to apply FFT filtering.
        :param nm_per_pixel: Scale of image in nanometers per pixel.
        """
        self.image_path = Path(image_path)
        self._image_source = self._read_image(self.image_path)
        self._image_grayscale_source = self._convert_to_grayscale(self._image_source)
        self.nm_per_pixel = nm_per_pixel or self._get_scale(self.image_path.with_suffix(".txt"))

        if cut_SEM:
            self._cut_image()

        self._image = self._image_source.copy()
        self._image_grayscale = self._image_grayscale_source.copy()

        if fft_filter:
            self._filter_image()
            self._update_RGB_image()

        self.segmenter = WatershedSegmenter(self._image_grayscale)

    def _get_scale(self, txt_path: Path | str) -> float | None:
        """
        Get image scale (nm per pixel) from a text file.

        :param txt_path: Path to the scale information file.
        :return: Computed scale or None if file not found.
        """
        try:
            with open(txt_path, "r", errors="ignore") as txt_file:
                for line in txt_file:
                    if line.startswith("$$SM_MICRON_BAR"):
                        pixels_per_bar = int(line.split()[1])
                    elif line.startswith("$$SM_MICRON_MARKER"):
                        size_per_bar = line.split()[1]
        except FileNotFoundError as e:
            print(f"Error reading scale information: {e}")
            nm_per_pixel = None
        else:
            unit = size_per_bar[-2:]
            size = float(size_per_bar[:-2])
            if unit == "nm":
                nm_per_bar = size
            elif unit == "um":
                nm_per_bar = size * 1e3  # convert to nanometers
            else:
                raise ValueError(f"Unknown unit in size_per_bar: {unit}")
            nm_per_pixel = nm_per_bar / pixels_per_bar
        return nm_per_pixel

    @plot_decorator
    def image(self) -> MatLike:
        """
        Return the current RGB image.

        :return: The RGB image.
        """
        self._update_RGB_image()
        return self._image

    @plot_decorator
    def image_grayscale(self) -> MatLike:
        """
        Return the current grayscale image.

        :return: The grayscale image.
        """
        return self._image_grayscale

    def _read_image(self, path: Path | str) -> MatLike:
        image = cv.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return image

    def _convert_to_grayscale(self, image: MatLike) -> MatLike:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def _convert_to_RGB(self, image: MatLike) -> MatLike:
        return cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    def _update_RGB_image(self) -> None:
        self._image = self._convert_to_RGB(self._image_grayscale)

    def _cut_image(self) -> None:
        self._image_source = self._image_source[:-128]
        self._image_grayscale_source = self._image_grayscale_source[:-128]

    def _filter_image(self, radius: int = 100, plot: bool = False) -> None:
        image = self._image_grayscale_source.copy()
        # Apply FFT
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        # Create a mask with a high-pass filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius * radius
        mask[mask_area] = 1

        # Apply mask and inverse FFT
        fshift_masked = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_masked)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        if plot:
            magnitude_spectrum = 20 * np.log(np.abs(np.fft.fftshift(f)) + 1)
            magnitude_spectrum = cv.normalize(magnitude_spectrum, magnitude_spectrum, 0, 255, cv.NORM_MINMAX)
            mask_circle_rgb = cv.circle(
                self._convert_to_RGB(magnitude_spectrum.astype(np.uint8)),
                (ccol, crow),
                radius,
                (255, 0, 0),
                thickness=5,
            )
            plt.figure()
            plt.imshow(mask_circle_rgb)
            plt.title(f"FFT Filter with radius {radius}")
            plt.axis("off")
            plt.show()

        self._image_grayscale = img_back.astype(np.uint8)

    def update_fft_radius(self, radius, plot_filter=False) -> None:
        """
        Update the FFT filter radius and refresh segmentation.

        :param radius: New FFT filter radius.
        :param plot_filter: Whether to display the FFT of the image with applied filter.
        """
        self._filter_image(radius, plot=plot_filter)
        self.segmenter = WatershedSegmenter(self._image_grayscale)  # update segmenter with new image

    def adjust_fft_mask(self) -> None:
        """
        Launch an interactive widget to adjust the FFT filter mask radius.
        """
        import ipywidgets

        def update_image(radius: int) -> None:
            self.update_fft_radius(radius, plot_filter=True)
            self.image_grayscale(plot=True)
            plt.title(f"FFT Filtered Image with radius {radius}")

        ipywidgets.interact(update_image, radius=(0, 250, 1))

    def get_diameters(self, in_nm: bool = True) -> NDArray[np.float64]:
        """
        Return the grain diameters.

        :param in_nm: Return values in nanometers if True, else in pixels.
        :return: Array of grain diameters.
        """
        diameters = np.array([cluster.feret_diameter_max for cluster in self.segmenter.clusters], dtype=np.float64)[1:]
        if in_nm and self.nm_per_pixel is not None:
            diameters *= self.nm_per_pixel
        return diameters

    def get_perimeters(self, in_nm: bool = True) -> NDArray[np.float64]:
        """
        Return the grain perimeters.

        :param in_nm: Return values in nanometers if True, else in pixels.
        :return: Array of grain perimeters.
        """
        perimeters = np.array([cluster.perimeter for cluster in self.segmenter.clusters], dtype=np.float64)[1:]
        if in_nm and self.nm_per_pixel is not None:
            perimeters *= self.nm_per_pixel
        return perimeters

    def get_areas(self, in_nm: bool = True) -> NDArray[np.float64]:
        """
        Return the grain areas.

        :param in_nm: Return values in nanometers squared if True, else in pixels squared.
        :return: Array of grain areas.
        """
        areas = np.array([cluster.area for cluster in self.segmenter.clusters], dtype=np.float64)[1:]
        if in_nm and self.nm_per_pixel is not None:
            areas *= (self.nm_per_pixel) ** 2
        return areas

    def _compute_stats(self, values: NDArray[np.float64]) -> dict[str, float]:
        return {
            "mean": np.mean(values).item(),
            "std": np.std(values).item(),
            "min": np.min(values).item(),
            "max": np.max(values).item(),
            "sum": np.sum(values).item(),
        }

    def get_stats(self, in_nm: bool = True) -> dict[str, dict[str, float]]:
        """
        Return statistical summaries (mean, std, min, max, sum) for diameters, perimeters, and areas.

        :param in_nm: Use nanometer units if True.
        :return: Dictionary of statistics per characteristic.
        """
        return {
            "diameters": self._compute_stats(self.get_diameters(in_nm)),
            "perimeters": self._compute_stats(self.get_perimeters(in_nm)),
            "areas": self._compute_stats(self.get_areas(in_nm)),
        }

    def save_results(self, path: Path | str = "results") -> None:
        """
        Save the processed images, statistics, and plots into the specified directory.

        :param path: Destination directory.
        """
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)

        cv.imwrite(str(path / "image.png"), self._image_grayscale)
        cv.imwrite(
            str(path / "image_with_markers.png"),
            cv.cvtColor(self.segmenter.marked_image(), cv.COLOR_RGB2BGR),
        )

        diameters = self.get_diameters()
        perimeters = self.get_perimeters()
        areas = self.get_areas()

        plotter = GrainPlotter(
            diameters=diameters,
            perimeters=perimeters,
            areas=areas,
            nm_per_pixel=self.nm_per_pixel,
        )

        plotter.plot_diameters(return_fig=True).savefig(path / "diameters.png", dpi=300)
        with open(path / "diameters.txt", "w") as f:
            f.write("\n".join(map(str, diameters)))

        plotter.plot_areas(return_fig=True).savefig(path / "areas.png", dpi=300)
        with open(path / "areas.txt", "w") as f:
            f.write("\n".join(map(str, areas)))

        with open(path / "diameters_fit.txt", "w") as f:
            x, pdf = plotter._lognorm_fit(diameters)
            f.write("\n".join(f"{x[i]}, {pdf[i]}" for i in range(len(x))))

        plotter.plot_perimeters(return_fig=True).savefig(path / "perimeters.png", dpi=300)
        with open(path / "perimeters.txt", "w") as f:
            f.write("\n".join(map(str, perimeters)))

        with open(path / "perimeters_fit.txt", "w") as f:
            x, pdf = plotter._lognorm_fit(perimeters)
            f.write("\n".join(f"{x[i]}, {pdf[i]}" for i in range(len(x))))

        with open(path / "stats.txt", "w") as f:
            stats = self.get_stats()
            for key, value in stats.items():
                f.write(f"{key}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"\t{subkey}: {subvalue}\n")

        plotter.plot_perimeters_vs_diameters(return_fig=True, fit=True).savefig(
            path / "perimeters_vs_diameters.png", dpi=300
        )

        plotter.plot_area_fractions(return_fig=True).savefig(path / "area_fractions.png", dpi=300)
        plotter.plot_area_fractions_vs_perimeter(return_fig=True).savefig(
            path / "area_fractions_vs_perimeter.png", dpi=300
        )
        with open(path / "area_fractions.txt", "w") as f:
            fractions = areas / areas.sum() * 100
            bins, hist = get_hist_data(data=diameters, nm_per_bin=1, quantile=0.995, weights=fractions)
            f.write("\n".join(f"{bins[i]}, {hist[i]}" for i in range(len(bins))))

        with open(path / "area_fractions_vs_perimeter.txt", "w") as f:
            bins, hist = get_hist_data(data=perimeters, nm_per_bin=3.5, quantile=0.995, weights=fractions)
            f.write("\n".join(f"{bins[i]}, {hist[i]}" for i in range(len(bins))))
            f.write("\n".join(f"{bins[i]}, {hist[i]}" for i in range(len(bins))))
