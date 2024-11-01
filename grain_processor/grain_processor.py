"""Based on the tutorial https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html"""

from functools import cached_property, wraps
from pathlib import Path

import cv2 as cv
import feret
import numpy as np
from matplotlib import pyplot as plt


def plot_decorator(func):
    @wraps(func)
    def wrapper(*args, plot=False, **kwargs):
        # call the original function to get the result
        result = func(*args, **kwargs)

        # if plot=True, plot the result
        if plot:
            plt.figure()
            plt.imshow(result, cmap="gray")
            plt.title(func.__name__[len("_") :].replace("_", " ").capitalize())
            plt.axis("off")
            plt.show()

        return result

    return wrapper


class GrainProcessor:
    def __init__(self, image_path: Path | str, cut_SEM=False, fft_filter=False):
        self.image_path = image_path
        self.cut_SEM = cut_SEM
        self.fft_filter = fft_filter

        self.image = self._image()
        self.image_grayscale = self._image(grayscale=True)

    @plot_decorator
    def _image(self, grayscale=False):
        if grayscale:
            image = cv.imread(self.image_path, cv.IMREAD_GRAYSCALE)
        else:
            image = cv.imread(self.image_path)

        if self.cut_SEM:
            image = image[:-128]

        if self.fft_filter:
            image = self._fft_filter(image)

        return image

    def _fft_filter(self, image, radius=100):
        # Convert image to grayscale if it is not already
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

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

        return img_back

    @plot_decorator
    def _threshold(self, blockSize=151):
        # blur to remove gaussian noise
        gray = cv.GaussianBlur(self.image_grayscale, (5, 5), cv.BORDER_DEFAULT)

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
        markers = cv.watershed(self.image, markers)

        return markers

    @plot_decorator
    def _image_non_contrast(self):
        markers = self._markers()
        kernel = np.ones((3, 3), np.uint8)
        img_no_contrast = self.image

        mask = (markers == -1).astype(np.uint8)
        dilated_mask = cv.dilate(mask, kernel, iterations=2)

        img_no_contrast[dilated_mask == 1] = [255, 0, 0]

        return img_no_contrast

    def collect_stat(self):
        areas = []
        feret_data = []
        markers = self._markers()
        for i in range(2, np.max(markers)):
            try:
                if np.average((markers == i)) == 0.0:
                    # print(f"no data {i}")
                    continue
                feret_data.append(feret.all(markers == i))
                areas.append(np.sum(markers == i))
            except IndexError:
                pass
                # print(i)
        feret_data_np = np.array(feret_data)
        pixel_sizes = np.sqrt(feret_data_np[:, 0] ** 2 + feret_data_np[:, 1] ** 2)
        all_areas = np.array(areas)
        nm_size = pixel_sizes * 2000 / 813  ## nm / pixels !!!!!!!!!!!!!!!!!!!!!
        return all_areas, nm_size

    @cached_property
    def merge_data_and_sort(self):
        all_areas, nm_size = self.collect_stat()

        merged_data = np.ones((all_areas.shape[0], 2))
        merged_data[:, 1] = all_areas
        merged_data[:, 0] = nm_size
        sorted_by_radius = merged_data[merged_data[:, 0].argsort()]
        return sorted_by_radius

    @cached_property
    def split_data_into_bins(self) -> list[np.array]:
        data = self.merge_data_and_sort
        delta_r = 2

        bins_n = int(np.ceil(np.max(data[:, 0]) / delta_r))
        answer = []
        for i in range(bins_n):
            answer.append(
                data[(data[:, 0] > i * delta_r) * (data[:, 0] < (i + 1) * delta_r)]
            )
        return answer

    @cached_property
    def get_plot_data(self, delta_r=2):
        data = self.merge_data_and_sort
        splited_data = self.split_data_into_bins
        total_area = np.sum(data[:, 1])
        x = [delta_r * (i + 1) for i in range(len(splited_data))]
        y = [np.sum(chunk[:, 1]) / total_area for chunk in splited_data]
        data = np.array([[i, j] for i, j in zip(x, y)])
        return data


# GP = GrainProcessor(r"C:\Users\Sergey\OneDrive\hzo grains\326\03.tif", fft_filter=True)
# GP.get_plot_data()
