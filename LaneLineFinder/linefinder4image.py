#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 22:04:12 2017

@author: alpha
"""

import cv2
import numpy as np

class LaneLineFinder4Image(object):

    def __init__(self, undist, linefilter, warper, fittool):
        self.undist = undist  # undistortion function
        self.lf = linefilter
        self.wp = warper
        self.ft = fittool
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.center_deviation = None

    def apply(self, img):
        img_undist = self.undist(img)
        img_binary = self.lf.apply(img_undist)
        img_binary_warped = self.wp.warp(img_binary)

        line_points = self.ft.locate_lines_by_blind_search(img_binary_warped)
        fits = self.ft.polyfit(*line_points)
        area_img = self.ft.fit_area_visualize(*fits)
        line_img = self.ft.line_points_visualize(*line_points)

        img_fit_warped = cv2.addWeighted(line_img, 1, area_img, 0.6, 0)
        img_fit = self.wp.warp_inv(img_fit_warped)
        out_img = cv2.addWeighted(img_undist, 0.8, img_fit, 0.3, 0)

        fit_l, fit_r = self.ft.polyfit(*line_points)
        info = self.ft.cal_rad_and_cendev(fit_l, fit_r)
        self.radius_of_curvature = np.mean(info[0])
        self.center_deviation = info[1]

        if self.radius_of_curvature <= 1500:
            message0 = 'Line curve radius is about {:.1f}m.'.format(self.radius_of_curvature)
        else:
            message0 = 'Road is nearly straight.'
        if self.center_deviation >= 0:
            message1 = 'Your vehicle is {:.2f}m left from road center.'.format(self.center_deviation)
        else:
            message1 = 'Your vehicle is {:.2f}m right from road center.'.format(np.abs(self.center_deviation))
        cv2.putText(out_img, message0, (200, 100), 0, 1.2, (255, 255, 0), 2)
        cv2.putText(out_img, message1, (200, 150), 0, 1.2, (255, 255, 0), 2)

        return out_img

if __name__ == '__main__':
    import os
    import glob
    import matplotlib.pyplot as plt
    from calibrate import CameraCalibrator
    from linefilter import LaneLineFilter
    from warp import WarpPerspective
    from fittool import LaneLineFitTool

    cc = CameraCalibrator()
    cc.load_matrix('../camera_mtx_dist.pk')
    undistort = cc.undistort

    lf = LaneLineFilter()
    wp = WarpPerspective()
    ft = LaneLineFitTool()
    linefinder = LaneLineFinder4Image(undistort, lf, wp, ft)

    img0 = plt.imread('../test_images/straight_lines2.jpg')
    result0 = linefinder.apply(img0)
    img1 = plt.imread('../test_images/test1.jpg')
    result1 = linefinder.apply(img1)

    plt.figure(figsize=(12, 7), dpi=100)
    plt.subplot(2, 2, 1)
    plt.imshow(img0, interpolation='bilinear')
    plt.title('Original Image')
    plt.subplot(2, 2, 2)
    plt.imshow(result0, interpolation='bilinear')
    plt.title('Result')
    plt.subplot(2, 2, 3)
    plt.imshow(img1, interpolation='bilinear')
    plt.subplot(2, 2, 4)
    plt.imshow(result1, interpolation='bilinear')
    plt.savefig('../pipline_image_example.jpg')

    test_img_paths = sorted(glob.glob('../test_images/*.jpg'))

    for test_img_path in test_img_paths:
        img = plt.imread(test_img_path)
        result = linefinder.apply(img)
        basename = os.path.basename(test_img_path)
        plt.imsave(os.path.join('../output_images',
                                basename.split('.')[0]+'_output.jpg'),
                   result, format='jpg')