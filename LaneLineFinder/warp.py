#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 19:16:59 2017

@author: alpha
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

class WarpPerspective(object):

    def __init__(self, src=None, dst=None, img_size=(1280, 720)):
        self.img_size = img_size
        if src is None or dst is None:
            self.src = np.float32(
                [[(img_size[0] / 2) - 68, img_size[1] / 2 + 90],
                 [0, img_size[1]],
                 [img_size[0], img_size[1]],
                 [(img_size[0] / 2) + 62, img_size[1] / 2 + 90]])
            self.dst = np.float32(
                [[img_size[0] / 8, 0],
                 [img_size[0] / 8, img_size[1]],
                 [img_size[0] * 7/8, img_size[1]],
                 [img_size[0] * 7/8, 0]])
        else:
            self.src = np.float32(src)
            self.dst = np.float32(dst)
        # Calculate transform matrix and inverse matrix
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp(self, img):
        '''Warp an image from perspective view to bird view.'''
        assert (img.shape[1], img.shape[0]) == self.img_size, 'Invalid image shape.'
        return cv2.warpPerspective(img, self.M, self.img_size)

    def warp_inv(self, img):
        '''Warp inversely an image from bird view to perspective view.'''
        assert (img.shape[1], img.shape[0]) == self.img_size, 'Invalid image shape.'
        return cv2.warpPerspective(img, self.M_inv, self.img_size)


if __name__ == '__main__':
    img = plt.imread('../output_images/straight_lines1_undist.jpg')
    wp = WarpPerspective()
    warped = wp.warp(img)
    plt.figure(figsize=(12, 4), dpi=100)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Undistorted Image')
    plt.subplot(1, 2, 2)
    plt.imshow(warped)
    plt.title('Warped Image')
    plt.savefig('../warp_example.jpg')