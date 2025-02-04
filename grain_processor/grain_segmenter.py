"""Based on the tutorial https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
Module providing watershed segmentation for grain images.
"""

import cv2 as cv
import numpy as np
import skimage.measure
import skimage.measure._regionprops
from cv2.typing import MatLike

from .utils import plot_decorator


class WatershedSegmenter:
    """
    Segment grain images using the watershed algorithm.

    This class applies adaptive thresholding, morphological operations, and watershed to segment grains.
    """

    THRESHOLD_BLOCK_SIZE: int = 151
    DISTANCE_TRANSFORM_THRESHOLD: float = 0.3

    def __init__(self, image: MatLike) -> None:
        """
        Initialize the segmenter with an image.

        :param image: The input image for segmentation.
        """
        self.image = image

    @plot_decorator
    def threshold(self) -> MatLike:
        """
        Apply adaptive thresholding to the blurred image.

        :return: The thresholded image.
        """
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
    def opening(self) -> MatLike:
        """
        Perform morphological opening to remove noise.

        :return: The opening of the image.
        """
        kernel = np.ones((3, 3), np.uint8)

        # apply erosion + dilation 2 times
        opening = cv.morphologyEx(self.threshold(), op=cv.MORPH_OPEN, kernel=kernel, iterations=2)

        return opening

    @plot_decorator
    def background(self) -> MatLike:
        """
        Determine the sure background by dilating the opened image.

        :return: The background image.
        """
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv.dilate(self.opening(), kernel, iterations=2)
        return sure_bg

    @plot_decorator
    def foreground(self) -> MatLike:
        """
        Extract the sure foreground areas using distance transform and thresholding.

        :return: The foreground image as a binary mask.
        """
        dist_transform = cv.distanceTransform(self.opening(), distanceType=cv.DIST_L2, maskSize=3)
        ret, sure_fg = cv.threshold(
            dist_transform,
            thresh=self.DISTANCE_TRANSFORM_THRESHOLD * dist_transform.max(),
            maxval=255,
            type=cv.THRESH_BINARY,
        )
        return sure_fg.astype(np.uint8)

    @plot_decorator
    def unknown_region(self) -> MatLike:
        """
        Compute the unknown region by subtracting the foreground from the background.

        :return: The unknown region as an image.
        """
        return cv.subtract(self.background(), self.foreground())

    @plot_decorator
    def markers(self) -> MatLike:
        """
        Generate markers for the watershed algorithm based on connected components.

        :return: Marker image for watershed segmentation.
        """
        blob_number, markers = cv.connectedComponents(self.foreground())
        markers = markers + 1
        markers[self.unknown_region() == 255] = 0
        if len(self.image.shape) == 2:
            image = cv.cvtColor(self.image, cv.COLOR_GRAY2RGB)
        markers = cv.watershed(image, markers)

        return markers

    @plot_decorator
    def marked_image(self) -> MatLike:
        """
        Overlay segmentation markers on the original image.

        :return: The image with markers indicating grain boundaries.
        """
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
        """
        Compute region properties (clusters) based on the watershed markers.

        :return: A list of properties for each grain.
        """
        return skimage.measure.regionprops(self.markers())
