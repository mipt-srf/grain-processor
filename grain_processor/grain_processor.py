"""Based on the tutorial https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html"""

from functools import cached_property, wraps
from pathlib import Path

import cv2 as cv
import numpy as np
import skimage.measure
from matplotlib import pyplot as plt
from scipy.stats import lognorm


def plot_decorator(func):
    @wraps(func)
    def wrapper(*args, plot=False, **kwargs):
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


class GrainProcessor:
    def __init__(
        self, image_path: Path | str, cut_SEM=False, fft_filter=False, scale=True
    ):
        self.image_path = Path(image_path)
        self._image_source = self._read_image(self.image_path)
        self._image_grayscale_source = self._convert_to_grayscale(self._image_source)

        if cut_SEM:
            self._cut_image()

        self._image = self._image_source.copy()
        self._image_grayscale = self._image_grayscale_source.copy()

        if fft_filter:
            self._filter_image()
            self._update_RGB_image()

        if scale:
            self._get_scale(self.image_path.with_suffix(".txt"))

    def _get_scale(self, txt_path: Path | str):
        try:
            with open(txt_path, "r") as file:
                for line in file:
                    if line.startswith("$$SM_MICRON_BAR"):
                        self.pixels_per_bar = int(line.split()[1])
                    elif line.startswith("$$SM_MICRON_MARKER"):
                        size_per_bar = line.split()[1]
        except FileNotFoundError as e:
            print(f"Error reading scale information: {e}")
            self.pixels_per_bar = None
            self.nanometers_per_bar = None
        else:
            unit = size_per_bar[-2:]
            size = float(size_per_bar[:-2])
            if unit == "nm":
                self.nanometers_per_bar = size
            elif unit == "um":
                self.nanometers_per_bar = size * 1e3  # convert to nanometers
            else:
                raise ValueError(f"Unknown unit in size_per_bar: {unit}")

    @plot_decorator
    def image(self):
        self._update_RGB_image()
        return self._image

    @plot_decorator
    def image_grayscale(self):
        return self._image_grayscale

    def _read_image(self, path: Path | str):
        image = cv.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return image

    def _convert_to_grayscale(self, image):
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def _convert_to_RGB(self, image):
        return cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    def _update_RGB_image(self):
        self._image = self._convert_to_RGB(self._image_grayscale)

    def _cut_image(self):
        self._image_source = self._image_source[:-128]
        self._image_grayscale_source = self._image_grayscale_source[:-128]

    def _filter_image(self, image=None, radius=100, plot=False):
        if image is None:
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

    def adjust_fft_mask(self):
        import ipywidgets

        def update_image(radius):
            self._image_grayscale = self._filter_image(radius, plot=True)
            plt.imshow(self._image_grayscale, cmap="gray")
            plt.title(f"FFT Filtered Image with radius {radius}")
            plt.axis("off")
            plt.show()

        ipywidgets.interact(update_image, radius=(0, 250, 1))

    @plot_decorator
    def _threshold(self, blockSize=151):
        # blur to remove gaussian noise
        gray = cv.GaussianBlur(self._image_grayscale, (5, 5), cv.BORDER_DEFAULT)

        # use adaptive thresholding to consider possible variations in brightness across the image
        thresh = cv.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv.THRESH_BINARY,
            blockSize=blockSize,
            C=0,
        )

        return thresh

    @plot_decorator
    def _opening(self):
        kernel = np.ones((3, 3), np.uint8)

        # apply erosion + dilation 2 times
        opening = cv.morphologyEx(
            self._threshold(), op=cv.MORPH_OPEN, kernel=kernel, iterations=2
        )

        return opening

    @plot_decorator
    def _background(self):
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv.dilate(self._opening(), kernel, iterations=2)
        return sure_bg

    @plot_decorator
    def _foreground(self):
        dist_transform = cv.distanceTransform(
            self._opening(), distanceType=cv.DIST_L2, maskSize=3
        )
        ret, sure_fg = cv.threshold(
            dist_transform,
            thresh=0.3 * dist_transform.max(),
            maxval=255,
            type=cv.THRESH_BINARY,
        )
        return sure_fg.astype(np.uint8)

    @plot_decorator
    def _unknown_region(self):
        return cv.subtract(self._background(), self._foreground())

    @plot_decorator
    def _markers(self):
        blob_number, markers = cv.connectedComponents(self._foreground())
        markers = markers + 1
        markers[self._unknown_region() == 255] = 0
        if len(self._image.shape) == 2:
            self._image = cv.cvtColor(self._image, cv.COLOR_GRAY2RGB)
        markers = cv.watershed(self._image, markers)

        return markers

    @plot_decorator
    def _image_non_contrast(self):
        markers = self._markers()
        kernel = np.ones((3, 3), np.uint8)
        img_no_contrast = self._image.copy()

        mask = (markers == -1).astype(np.uint8)
        dilated_mask = cv.dilate(mask, kernel, iterations=2)

        img_no_contrast[dilated_mask == 1] = [0, 0, 255]

        return img_no_contrast

    @cached_property
    def clusters(self):
        return skimage.measure.regionprops(self._markers())

    def get_diameters(self):
        return np.array([cluster.feret_diameter_max for cluster in self.clusters])[1:]

    def get_areas(self):
        return np.array([cluster.area for cluster in self.clusters])[1:]

    def _lognorm_fit(self, data):
        shape, loc, scale = lognorm.fit(data, floc=0)
        x = np.linspace(0, np.quantile(data, 0.99), 100)
        pdf = lognorm.pdf(x, shape, loc, scale)

        return x, pdf

    def _plot_distribution(self, data, xlabel, probability=True, fit=True):
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
            )  # sum of all bins is 100%
            plt.ylabel("Probability, %")

        plt.show()

    def plot_diameters(self, fit=True, probability=True, return_fig=False):
        fig = plt.figure()
        diameters = self.get_diameters()

        label = "Grain diameter, "
        if self.pixels_per_bar is not None:
            diameters *= self.nanometers_per_bar / self.pixels_per_bar
            label += "nm"
        else:
            label += "px"

        self._plot_distribution(
            data=diameters, xlabel=label, probability=probability, fit=fit
        )
        if return_fig:
            return fig

    def plot_areas(self, fit=False, probability=True, return_fig=False):
        fig = plt.figure()
        areas = self.get_areas()

        label = r"Grain area, "
        if self.pixels_per_bar is not None:
            areas *= (self.nanometers_per_bar / self.pixels_per_bar) ** 2
            label += "nm$^2$"
        else:
            label += "px$^2$"

        self._plot_distribution(
            data=areas, xlabel=label, probability=probability, fit=fit
        )
        if return_fig:
            return fig

    def save_results(self, path: Path | str = "results"):
        path = Path(path)
        path.mkdir(exist_ok=True)

        cv.imwrite(path / "image.png", self._image_grayscale)
        cv.imwrite(path / "image_with_markers.png", self._image_non_contrast())

        self.plot_diameters(return_fig=True).savefig(path / "diameters.png", dpi=300)
        with open(path / "diameters.txt", "w") as f:
            f.write("\n".join(map(str, self.get_diameters())))

        self.plot_areas(return_fig=True).savefig(path / "areas.png", dpi=300)
        with open(path / "areas.txt", "w") as f:
            f.write("\n".join(map(str, self.get_areas())))

        with open(path / "diameters_fit.txt", "w") as f:
            x, pdf = self._lognorm_fit(self.get_diameters())
            f.write("\n".join(f"{x[i]}, {pdf[i]}" for i in range(len(x))))
