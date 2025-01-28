from pathlib import Path

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from .grain_plotter import GrainPlotter
from .grain_segmenter import WatershedSegmenter
from .utils import plot_decorator


class GrainProcessor:
    def __init__(
        self,
        image_path: Path | str,
        cut_SEM: bool = False,
        fft_filter: bool = False,
        nm_per_pixel: float | None = None,
    ) -> None:
        self.image_path = Path(image_path)
        self._image_source = self._read_image(self.image_path)
        self._image_grayscale_source = self._convert_to_grayscale(self._image_source)
        self.nm_per_pixel = nm_per_pixel or self._get_scale(
            self.image_path.with_suffix(".txt")
        )

        if cut_SEM:
            self._cut_image()

        self._image = self._image_source.copy()
        self._image_grayscale = self._image_grayscale_source.copy()

        if fft_filter:
            self._filter_image()
            self._update_RGB_image()

        self.segmenter = WatershedSegmenter(self._image_grayscale)

    def _get_scale(self, txt_path: Path | str) -> float | None:
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
    def image(self) -> np.ndarray:
        self._update_RGB_image()
        return self._image

    @plot_decorator
    def image_grayscale(self) -> np.ndarray:
        return self._image_grayscale

    def _read_image(self, path: Path | str) -> np.ndarray:
        image = cv.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return image

    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def _convert_to_RGB(self, image: np.ndarray) -> np.ndarray:
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
            magnitude_spectrum = cv.normalize(
                magnitude_spectrum, None, 0, 255, cv.NORM_MINMAX
            )
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

    def adjust_fft_mask(self) -> None:
        import ipywidgets

        def update_image(radius):
            self._filter_image(radius, plot=True)
            self.segmenter = WatershedSegmenter(
                self._image_grayscale
            )  # update segmenter with new image
            plt.imshow(self._image_grayscale, cmap="gray")
            plt.title(f"FFT Filtered Image with radius {radius}")
            plt.axis("off")
            plt.show()

        ipywidgets.interact(update_image, radius=(0, 250, 1))

    def get_diameters(self, in_nm: bool = True) -> np.ndarray:
        diameters = np.array(
            [cluster.feret_diameter_max for cluster in self.segmenter.clusters]
        )[1:]
        if in_nm and self.nm_per_pixel is not None:
            diameters *= self.nm_per_pixel
        return diameters

    def get_perimeters(self, in_nm: bool = True) -> np.ndarray:
        perimeters = np.array(
            [cluster.perimeter for cluster in self.segmenter.clusters]
        )[1:]
        if in_nm and self.nm_per_pixel is not None:
            perimeters *= self.nm_per_pixel
        return perimeters

    def get_areas(self, in_nm: bool = True) -> np.ndarray:
        areas = np.array([cluster.area for cluster in self.segmenter.clusters])[1:]
        if in_nm and self.nm_per_pixel is not None:
            areas *= (self.nm_per_pixel) ** 2
        return areas

    def get_stats(self, in_nm: bool = True) -> dict[str, dict[str, float]]:
        diameters = self.get_diameters(in_nm)
        perimeters = self.get_perimeters(in_nm)
        areas = self.get_areas(in_nm)

        return {
            "diameters": {
                "mean": np.mean(diameters),
                "std": np.std(diameters),
                "min": np.min(diameters),
                "max": np.max(diameters),
                "sum": np.sum(diameters),
            },
            "perimeters": {
                "mean": np.mean(perimeters),
                "std": np.std(perimeters),
                "min": np.min(perimeters),
                "max": np.max(perimeters),
                "sum": np.sum(perimeters),
            },
            "areas": {
                "mean": np.mean(areas),
                "std": np.std(areas),
                "min": np.min(areas),
                "max": np.max(areas),
                "sum": np.sum(areas),
            },
        }

    def save_results(self, path: Path | str = "results") -> None:
        path = Path(path)
        path.mkdir(exist_ok=True)

        cv.imwrite(path / "image.png", self._image_grayscale)
        cv.imwrite(
            path / "image_with_markers.png",
            self.segmenter.marked_image().astype(np.uint8),
        )

        plotter = GrainPlotter(
            diameters=self.get_diameters(),
            perimeters=self.get_perimeters(),
            areas=self.get_areas(),
            nm_per_pixel=self.nm_per_pixel,
        )

        plotter.plot_diameters(return_fig=True).savefig(path / "diameters.png", dpi=300)
        with open(path / "diameters.txt", "w") as f:
            f.write("\n".join(map(str, self.get_diameters())))

        plotter.plot_areas(return_fig=True).savefig(path / "areas.png", dpi=300)
        with open(path / "areas.txt", "w") as f:
            f.write("\n".join(map(str, self.get_areas())))

        with open(path / "diameters_fit.txt", "w") as f:
            x, pdf = plotter._lognorm_fit(self.get_diameters())
            f.write("\n".join(f"{x[i]}, {pdf[i]}" for i in range(len(x))))

        plotter.plot_perimeters(return_fig=True).savefig(
            path / "perimeters.png", dpi=300
        )
        with open(path / "perimeters.txt", "w") as f:
            f.write("\n".join(map(str, self.get_perimeters())))

        with open(path / "perimeters_fit.txt", "w") as f:
            x, pdf = plotter._lognorm_fit(self.get_perimeters())
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

        plotter.plot_area_fractions(return_fig=True).savefig(
            path / "area_fractions.png", dpi=300
        )
        plotter.plot_area_fractions_vs_perimeter(return_fig=True).savefig(
            path / "area_fractions_vs_perimeter.png", dpi=300
        )
