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
        self.image_path = image_path
        self.__image_source = self.__read_image(self.image_path)
        self.__image_grayscale_source = self._convert_to_grayscale(self.__image_source)

        self.cut_SEM = cut_SEM
        self.fft_filter = fft_filter

        if cut_SEM:
            self.__cut_image()

        self._image = self.__image_source
        self._image_grayscale = self.__image_grayscale_source

        if self.fft_filter:
            self._image_grayscale = self.__filter_image(self.__image_grayscale_source)
            self.__update_RGB_image()

    @plot_decorator
    def image(self):
        self.__update_RGB_image()
        return self._image

    @plot_decorator
    def image_grayscale(self):
        return self._image_grayscale

    def __read_image(self, path: Path | str):
        return cv.imread(self.image_path)

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

        img_no_contrast[dilated_mask == 1] = [255, 0, 0]

        return img_no_contrast

    @cached_property
    def clusters(self):
        return skimage.measure.regionprops(self._markers())

    def get_diameters(self, plot=False, fit=True):
        diameters = np.array([cluster.feret_diameter_max for cluster in self.clusters])[
            1:
        ]

        if plot:
            max_diameter = np.quantile(diameters, 0.9) * 1.1
            bins = np.linspace(0, max_diameter, 50)
            plt.hist(diameters, bins=bins, density=False, color="teal", alpha=0.6)
            plt.xlabel("Grain diameter, px")
            plt.ylabel("Count")

            if fit:
                shape, loc, scale = lognorm.fit(diameters, floc=0)
                x = np.linspace(diameters.min(), max_diameter, 100)
                pdf = lognorm.pdf(x, shape, loc, scale)

                # Scale the PDF to match the histogram counts
                bin_width = bins[1] - bins[0]
                pdf_scaled = pdf * len(diameters) * bin_width

                plt.plot(x, pdf_scaled, "r-", lw=2, color="lightcoral")
                plt.xlim(0, max_diameter)
            plt.show()

        return diameters

    def get_areas(self, plot=False, fit=True):
        areas = np.array([cluster.area for cluster in self.clusters])[1:]

        if plot:
            area_quantile = np.quantile(areas, 0.9) * 1.1
            bins = np.linspace(0, area_quantile, 50)
            plt.hist(areas, bins=bins, density=False, color="teal", alpha=0.6)
            plt.xlabel("Grain area, px")
            plt.ylabel("Count")

            if fit:
                shape, loc, scale = lognorm.fit(areas, floc=0)
                x = np.linspace(areas.min(), area_quantile, 100)
                pdf = lognorm.pdf(x, shape, loc, scale)

                # Scale the PDF to match the histogram counts
                # bin_width = bins[1] - bins[0]
                # pdf_scaled = pdf * len(areas) * bin_width

                plt.plot(x, pdf, "r-", lw=2)
                plt.xlim(0, area_quantile)
            plt.show()

        return areas
