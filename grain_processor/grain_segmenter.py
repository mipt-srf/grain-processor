"""Based on the tutorial https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html"""

import cv2 as cv
import numpy as np
import skimage.measure
import skimage.measure._regionprops

from .utils import plot_decorator


class WatershedSegmenter:
    THRESHOLD_BLOCK_SIZE = 151
    DISTANCE_TRANSFORM_THRESHOLD = 0.3

    def __init__(self, image: np.ndarray) -> None:
        self.image = image

    @plot_decorator
    def threshold(self) -> np.ndarray:
        # blur to remove gaussian noise
        gray = cv.GaussianBlur(self.image, (5, 5), cv.BORDER_DEFAULT)

        # use adaptive thresholding to consider possible variations in brightness across the image
        thresh = cv.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv.THRESH_BINARY,
            blockSize=self.THRESHOLD_BLOCK_SIZE,
            C=0,
        )

        return thresh

    @plot_decorator
    def opening(self) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)

        # apply erosion + dilation 2 times
        opening = cv.morphologyEx(
            self.threshold(), op=cv.MORPH_OPEN, kernel=kernel, iterations=2
        )

        return opening

    @plot_decorator
    def background(self) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv.dilate(self.opening(), kernel, iterations=2)
        return sure_bg

    @plot_decorator
    def foreground(self) -> np.ndarray:
        dist_transform = cv.distanceTransform(
            self.opening(), distanceType=cv.DIST_L2, maskSize=3
        )
        ret, sure_fg = cv.threshold(
            dist_transform,
            thresh=self.DISTANCE_TRANSFORM_THRESHOLD * dist_transform.max(),
            maxval=255,
            type=cv.THRESH_BINARY,
        )
        return sure_fg.astype(np.uint8)

    @plot_decorator
    def unknown_region(self) -> np.ndarray:
        return cv.subtract(self.background(), self.foreground())

    @plot_decorator
    def markers(self) -> np.ndarray:
        blob_number, markers = cv.connectedComponents(self.foreground())
        markers = markers + 1
        markers[self.unknown_region() == 255] = 0
        if len(self.image.shape) == 2:
            image = cv.cvtColor(self.image, cv.COLOR_GRAY2RGB)
        markers = cv.watershed(image, markers)

        return markers

    @plot_decorator
    def marked_image(self) -> np.ndarray:
        markers = self.markers()
        kernel = np.ones((3, 3), np.uint8)

        mask = (markers == -1).astype(np.uint8)
        dilated_mask = cv.dilate(mask, kernel, iterations=2)

        if len(self.image.shape) == 2:
            marked_image = cv.cvtColor(self.image, cv.COLOR_GRAY2RGB)
        marked_image[dilated_mask == 1] = [255, 0, 0]

        return marked_image

    @property
    def clusters(self) -> list[skimage.measure._regionprops.RegionProperties]:
        return skimage.measure.regionprops(self.markers())
