#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 19:19:02 2017

@author: alpha
"""

import cv2
import numpy as np

class LaneLineFitTool(object):
    '''A collection of methods doing line fitting.'''
    def __init__(self, img_size=(1280, 720), margin=85, area_color=(0, 255, 0),
                 color_left=(255, 0, 0), color_right=(0, 0, 255),
                 xm_per_pix=3.8/700, ym_per_pix=32/720):
        self.img_size = img_size
        self.margin = margin
        self.area_color = area_color
        self.color_left = color_left
        self.color_right = color_right
        self.margin_points = {'l': [], 'r': []}
        # meters per pixel in x, y dimension, based on 2 meter
        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix

    def polyfit(self, xl, yl, xr, yr):
        fit_l = np.polyfit(yl, xl, 2)
        fit_r = np.polyfit(yr, xr, 2)
        return fit_l, fit_r

    def line_points_visualize(self, xl, yl, xr, yr, draw_margin=False):
        '''Visualize the located line pixels'''
        shape_x, shape_y = self.img_size
        line_img = np.zeros((shape_y, shape_x, 3)).astype(np.uint8)
        line_img[yl, xl] = list(self.color_left)
        line_img[yr, xr] = list(self.color_right)

        if draw_margin:
            margin_points_l = self.margin_points['l']
            margin_points_r = self.margin_points['r']
            if margin_points_l:
                mcolor_l = np.array(self.color_left) + 127
                mcolor_r = np.array(self.color_right) + 127
                mcolor_l[mcolor_l > 255] = 255
                mcolor_r[mcolor_r > 255] = 255
                mcolor_l = tuple(mcolor_l.tolist())
                mcolor_r = tuple(mcolor_r.tolist())
                cv2.polylines(line_img, np.int_([margin_points_l]), True, mcolor_l, 2, cv2.LINE_AA)
                cv2.polylines(line_img, np.int_([margin_points_r]), True, mcolor_r, 2, cv2.LINE_AA)
            else:
                print('No margin points defined!')

        return line_img

    def fit_area_visualize(self, fit_l, fit_r):
        '''Visualize the area between left and right fit lines'''
        shape_x, shape_y = self.img_size
        y_points = np.linspace(0, shape_y - 1, shape_y)
        xfit_l_points = fit_l[0] * y_points ** 2 + fit_l[1] * y_points + fit_l[2]
        xfit_r_points = fit_r[0] * y_points ** 2 + fit_r[1] * y_points + fit_r[2]
        fit_l_points = list(zip(xfit_l_points, y_points))
        fit_r_points = list(zip(xfit_r_points, y_points))
        fit_r_points.reverse()
        fit_points = fit_l_points + fit_r_points
        area_img = np.zeros((shape_y, shape_x, 3)).astype(np.uint8)
        cv2.fillPoly(area_img, np.int_([fit_points]), self.area_color)
        return area_img

    def locate_lines_by_blind_search(self, img_binary_warped, nwindows=9, minpix=50,
                                     make_margin_points=False):
        '''
        Blindly search line pixels from an warped binary image.
        Return located lane line pixel indices.
        '''
        shape_y, shape_x = img_binary_warped.shape[:2]
        assert (shape_x,shape_y) == self.img_size, 'Invalid shape.'
        # Histgram of nonzero pixels along y for the bottom part of the image.
        hist = np.sum(img_binary_warped[np.int(shape_y / 2):, :], axis=0)
        # Set base line position.
        midpoint = np.int(shape_x / 2)
        margin = self.margin
        xl_c_base = np.argmax(hist[:midpoint])
        xr_c_base = np.argmax(hist[midpoint:]) + midpoint
        win_height = np.int(shape_y / nwindows)
        xl_c = xl_c_base
        xr_c = xr_c_base
        lane_idxs_l = []
        lane_idxs_r = []
        nonzero = img_binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if make_margin_points:
            margin_points_l_l = []
            margin_points_l_r = []
            margin_points_r_l = []
            margin_points_r_r = []

        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            y_u = img_binary_warped.shape[0] - (window + 1) * win_height
            y_d = img_binary_warped.shape[0] - window * win_height
            xl_l = xl_c - margin
            xl_r = xl_c + margin
            xr_l = xr_c - margin
            xr_r = xr_c + margin
            # Append margin points
            if make_margin_points:
                margin_points_l_l.append((xl_l, y_d))
                margin_points_l_l.append((xl_l, y_u))
                margin_points_l_r.append((xl_r, y_d))
                margin_points_l_r.append((xl_r, y_u))
                margin_points_r_l.append((xr_l, y_d))
                margin_points_r_l.append((xr_l, y_u))
                margin_points_r_r.append((xr_r, y_d))
                margin_points_r_r.append((xr_r, y_u))
            # Identify the nonzero pixels in x and y within the window
            good_idxs_l = ((nonzeroy >= y_u) & (nonzeroy < y_d) & (nonzerox >= xl_l) & (nonzerox < xl_r)).nonzero()[0]
            good_idxs_r = ((nonzeroy >= y_u) & (nonzeroy < y_d) & (nonzerox >= xr_l) & (nonzerox < xr_r)).nonzero()[0]
            # Append these indices to the lists
            lane_idxs_l.append(good_idxs_l)
            lane_idxs_r.append(good_idxs_r)
            # If found > minpix pixels, recenter next window on their mean position
            if len(good_idxs_l) > minpix:
                xl_c = np.int(np.median(nonzerox[good_idxs_l]))
            if len(good_idxs_r) > minpix:
                xr_c = np.int(np.median(nonzerox[good_idxs_r]))

        if make_margin_points:
            margin_points_l_r.reverse()
            margin_points_r_r.reverse()
            self.margin_points['l'] = margin_points_l_l + margin_points_l_r
            self.margin_points['r'] = margin_points_r_l + margin_points_r_r

        lane_idxs_l = np.concatenate(lane_idxs_l)
        lane_idxs_r = np.concatenate(lane_idxs_r)

        xl = nonzerox[lane_idxs_l]
        yl = nonzeroy[lane_idxs_l]
        xr = nonzerox[lane_idxs_r]
        yr = nonzeroy[lane_idxs_r]

        return xl, yl, xr, yr

    def locate_lines_by_fit_margin(self, img_binary_warped, fit_l, fit_r,
                                   make_margin_points=False):
        '''
        Search line pixels from an warped binary image using previous fitting params with a margin.
        Return located lane line pixel indices.
        '''
        shape_y, shape_x = img_binary_warped.shape[:2]
        assert (shape_x,shape_y) == self.img_size, 'Invalid shape.'
        margin = self.margin

        if make_margin_points:
            y_points = np.linspace(0, shape_y - 1, shape_y)
            xfit_l_points = fit_l[0] * y_points ** 2 + fit_l[1] * y_points + fit_l[2]
            xfit_r_points = fit_r[0] * y_points ** 2 + fit_r[1] * y_points + fit_r[2]
            margin_points_l_l = list(zip(xfit_l_points - margin, y_points))
            margin_points_l_r = list(zip(xfit_l_points + margin, y_points))
            margin_points_l_r.reverse()
            margin_points_r_l = list(zip(xfit_r_points - margin, y_points))
            margin_points_r_r = list(zip(xfit_r_points + margin, y_points))
            margin_points_r_r.reverse()
            self.margin_points['l'] = margin_points_l_l + margin_points_l_r
            self.margin_points['r'] = margin_points_r_l + margin_points_r_r

        nonzero = img_binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        xfity_l_points = fit_l[0] * nonzeroy ** 2 + fit_l[1] * nonzeroy + fit_l[2]
        xfity_r_points = fit_r[0] * nonzeroy ** 2 + fit_r[1] * nonzeroy + fit_r[2]

        lane_idxs_l = ((nonzerox > (xfity_l_points - margin)) & (nonzerox < (xfity_l_points + margin)))
        lane_idxs_r = ((nonzerox > (xfity_r_points - margin)) & (nonzerox < (xfity_r_points + margin)))

        xl = nonzerox[lane_idxs_l]
        yl = nonzeroy[lane_idxs_l]
        xr = nonzerox[lane_idxs_r]
        yr = nonzeroy[lane_idxs_r]

        return xl, yl, xr, yr

    def cal_rad_and_cendev(self, fit_l, fit_r):
        shape_y = self.img_size[1]
        y_points = np.linspace(0, shape_y - 1, shape_y)
        xfit_l_points = fit_l[0] * y_points ** 2 + fit_l[1] * y_points + fit_l[2]
        xfit_r_points = fit_r[0] * y_points ** 2 + fit_r[1] * y_points + fit_r[2]
        cendev = (np.mean([xfit_l_points[-1], xfit_r_points[-1]]) - self.img_size[0] / 2 + 1) * self.xm_per_pix
        # Fit new polynomials to x,y in real world space
        ym_points = y_points * self.ym_per_pix
        mfit_l = np.polyfit(ym_points, xfit_l_points * self.xm_per_pix, 2)
        mfit_r = np.polyfit(ym_points, xfit_r_points * self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        curverad_l = ((1 + (2*mfit_l[0]*np.max(ym_points) + mfit_l[1])**2)**1.5) / np.abs(2*mfit_l[0])
        curverad_r = ((1 + (2*mfit_r[0]*np.max(ym_points) + mfit_r[1])**2)**1.5) / np.abs(2*mfit_r[0])
        return (curverad_l, curverad_r), cendev

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from calibrate import CameraCalibrator
    from linefilter import LaneLineFilter
    from warp import WarpPerspective

    img = plt.imread('../test_images/test5.jpg')

    # Undistortion
    cc = CameraCalibrator()
    cc.load_matrix(fname='../camera_mtx_dist.pk')
    undist_img = cc.undistort(img)

    # Binary filter
    llf = LaneLineFilter()
    binary = llf.apply(undist_img)

    # Perspective Warp
    wp = WarpPerspective()
    binary_warped = wp.warp(binary)

    plt.figure(figsize=(18, 7.5), dpi=100)
    plt.subplot(2, 3, 1)
    plt.imshow(undist_img)
    plt.title('Undistorted Image')
    plt.subplot(2, 3, 2)
    plt.imshow(binary, 'gray')
    plt.title('Binary Image')
    plt.subplot(2, 3, 3)
    plt.imshow(binary_warped, 'gray')
    plt.title('Binary Warped')

    # Locate line pixels by blind search method
    ft = LaneLineFitTool()
    line_points = ft.locate_lines_by_blind_search(binary_warped, make_margin_points=True)
    fits = ft.polyfit(*line_points)
    line_img = ft.line_points_visualize(*line_points, draw_margin=True)
    area_img = ft.fit_area_visualize(*fits)
    img_fit_warped = cv2.addWeighted(line_img, 1, area_img, 0.6, 0)

    plt.subplot(2, 3, 4)
    plt.imshow(img_fit_warped)
    plt.title('Fitting Result (Blind Search)')

    # Locate line pixels by fit margin.
    line_points = ft.locate_lines_by_fit_margin(binary_warped, *fits, make_margin_points=True)
    fits = ft.polyfit(*line_points)
    line_img = ft.line_points_visualize(*line_points, draw_margin=True)
    area_img = ft.fit_area_visualize(*fits)
    img_fit_warped = cv2.addWeighted(line_img, 1, area_img, 0.6, 0)

    plt.subplot(2, 3, 5)
    plt.imshow(img_fit_warped)
    plt.title('Fitting Result (Fit Margin)')

    # Unwarp the fitting result to undistorted image.
    img_fit = wp.warp_inv(img_fit_warped)
    out_img = cv2.addWeighted(undist_img, 0.8, img_fit, 0.3, 0)

    plt.subplot(2, 3, 6)
    plt.imshow(out_img)
    plt.title('Result')

    plt.savefig('../fitting_example.jpg')