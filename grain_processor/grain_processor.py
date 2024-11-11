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
    def __init__(self, image_path: Path | str, cut_SEM=False, fft_filter=False):
        self.image_path = Path(image_path)
        self.__image_source = self.__read_image(self.image_path)
        self.__image_grayscale_source = self._convert_to_grayscale(self.__image_source)

        self.cut_SEM = cut_SEM
        self.fft_filter = fft_filter

        if cut_SEM:
            self.__cut_image()

        if scale:
            self.__get_scale(self.image_path.with_suffix(".txt"))

        self._image = self.__image_source
        self._image_grayscale = self.__image_grayscale_source

        if self.fft_filter:
            self._image_grayscale = self.__filter_image(self.__image_grayscale_source)
            self.__update_RGB_image()

    def __get_scale(self, txt_path: Path | str):
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
        self.__update_RGB_image()
        return self._image

    @plot_decorator
    def image_grayscale(self):
        return self._image_grayscale

    def __read_image(self, path: Path | str):
        image = cv.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return image

    def _convert_to_grayscale(self, image):
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def __convert_to_RGB(self, image):
        return cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    def __update_RGB_image(self):
        self._image = self.__convert_to_RGB(self._image_grayscale)

    def __cut_image(self):
        self.__image_source = self.__image_source[:-128]
        self.__image_grayscale_source = self.__image_grayscale_source[:-128]

    def __filter_image(self, image, radius=100):
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
        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return img_back.astype(np.uint8)

    def adjust_fft_mask(self):
        import ipywidgets

        def update_image(radius):
            self._image_grayscale = self.__filter_image(
                self.__image_grayscale_source, radius
            )
            plt.imshow(self._image_grayscale, cmap="gray")
            plt.title(f"FFT Filtered Image with Radius {radius}")
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

    def __lognorm_fit(self, data):
        shape, loc, scale = lognorm.fit(data, floc=0)
        x = np.linspace(data.min(), data.max(), 100)
        pdf = lognorm.pdf(x, shape, loc, scale)

        return x, pdf

    def __plot_distribution(self, data, xlabel, ylabel, fit=True):
        max_data = np.quantile(data, 0.95) * 1.3
        bins = np.linspace(0, max_data, 50)
        plt.hist(data, bins=bins, color="teal", alpha=0.6)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if fit:
            x, pdf = self.__lognorm_fit(data)

            bin_width = bins[1] - bins[0]
            pdf_scaled = pdf * len(data) * bin_width

            plt.plot(x, pdf_scaled, "r-", lw=2, color="lightcoral")
            plt.xlim(0, max_data)
        plt.show()

    def plot_diameters(self, fit=True, return_fig=False):
        fig = plt.figure()
        diameters = self.get_diameters()
        self.__plot_distribution(diameters, "Grain diameter, px", "Count", fit)
        if return_fig:
            return fig

    def plot_areas(self, fit=False, return_fig=False):
        fig = plt.figure()
        areas = self.get_areas()
        self.__plot_distribution(areas, "Grain area, px^2", "Count", fit)
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
            x, pdf = self.__lognorm_fit(self.get_diameters())
            f.write("\n".join(f"{x[i]}, {pdf[i]}" for i in range(len(x))))
