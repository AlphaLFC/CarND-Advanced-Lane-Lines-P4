#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 20:29:12 2017

@author: alpha
"""

import cv2
import numpy as np


class LaneLineFilter(object):
    '''
    A binary filter for lane lines.
    '''
    def __init__(self, sobel_thresh=(20, 100), sobel_kernel=3, sm_kernel=3,
                 l_thresh=170, b_thresh=60):
        self.sobel_thresh = sobel_thresh
        self.sobel_kernel = sobel_kernel
        self.sm_kernel = sm_kernel  # Gaussian smooth kernel size
        # The thresholds of the scaled and median-subtracted L & B channels of LAB color space image.
        self.l_thresh = l_thresh
        self.b_thresh = b_thresh

    def _lab_thresh(self, img):
        '''
        Input should be a RGB image array.
        Return is a processed binary of the LAB color space image.
        '''
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0]
        B = lab[:, :, 2]
        # Median filtering
        median_L = np.int(np.median(L))
        median_B = np.int(np.median(B))
        L[L < median_L] = median_L
        B[B < median_B] = median_B
        # Calculate line profile gradients for L channel
        sx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=self.sobel_kernel)
        sy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=self.sobel_kernel)
        abs_sx = np.abs(sx)
        abs_sy = np.abs(sy)
        scaled_sx = np.uint8(255 * abs_sx / np.max(abs_sx))
        scaled_sy = np.uint8(255 * abs_sy / np.max(abs_sy))
        # Gaussian smoothing to reduce noises
        scaled_sx = cv2.GaussianBlur(scaled_sx, (self.sm_kernel,)*2, self.sm_kernel)
        scaled_sy = cv2.GaussianBlur(scaled_sy, (self.sm_kernel,)*2, self.sm_kernel)
        # Sobel thresh
        binary = np.zeros_like(scaled_sx)
        min_thresh, max_thresh = self.sobel_thresh
        binary[(scaled_sx > min_thresh) & (scaled_sx <= max_thresh) & \
               (scaled_sy > min_thresh) & (scaled_sy <= max_thresh)] = 1
        # L channel thresh for white color extraction
        # B channel thresh for yellow color extraction
        l_scaled = (L - np.min(L)) * 255.0 / (np.max(L) - np.min(L))
        b_scaled = (B - np.min(B)) * 255.0 / (np.max(B) - np.min(B))
        binary[l_scaled > self.l_thresh] = 1
        binary[b_scaled > self.b_thresh] = 1
        return binary

    def apply(self, img):
        return self._lab_thresh(img)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = plt.imread('../test_images/test5.jpg')
    llf = LaneLineFilter()
    binary = llf.apply(img)
    plt.figure(figsize=(12, 4), dpi=100)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(binary, 'gray')
    plt.title('Binary Image')
    plt.savefig('../binary_example.jpg')