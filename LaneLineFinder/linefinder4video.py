#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 22:12:00 2017

@author: alpha
"""

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from collections import deque

from linefinder4image import LaneLineFinder4Image


class LaneLineFinder4Video(LaneLineFinder4Image):

    def __init__(self, n_recs=10, **kargs):
        super(LaneLineFinder4Video, self).__init__(**kargs)
        # the number of last records to be used.
        self.n_recs = n_recs

        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        self.n_rads = deque(maxlen=self.n_recs)
        # distance in meters of vehicle center from the line
        self.center_deviation = None
        self.n_cendevs = deque(maxlen=self.n_recs)

        self.use_fit_margin = False

        self.n_good_fit = deque(maxlen=self.n_recs)
        self.n_good_fit_l = deque(maxlen=self.n_recs)
        self.n_good_fit_r = deque(maxlen=self.n_recs)

        # polynomial coefficients of last n fits
        self.n_fits_l = deque(maxlen=self.n_recs)
        self.n_fits_r = deque(maxlen=self.n_recs)
        # polynomial coefficients averaged over the last n iterations
        self.ave_fit_l = np.float32([0, 0, 300])
        self.ave_fit_r = np.float32([0, 0, 980])

        self.line_points = None

    def apply(self, img):
        img_undist = self.undist(img)
        img_binary = self.lf.apply(img_undist)
        img_binary_warped = self.wp.warp(img_binary)

        if self.use_fit_margin:
            self.line_points = self.ft.locate_lines_by_fit_margin(img_binary_warped,
                                                                  self.ave_fit_l, self.ave_fit_r)
        else:
            self.line_points = self.ft.locate_lines_by_blind_search(img_binary_warped)

        self.sanity_check_fit()

        area_img = self.ft.fit_area_visualize(self.ave_fit_l, self.ave_fit_r)
        line_img = self.ft.line_points_visualize(*self.line_points)

        img_fit_warped = cv2.addWeighted(line_img, 1, area_img, 0.6, 0)
        img_fit = self.wp.warp_inv(img_fit_warped)
        out_img = cv2.addWeighted(img_undist, 0.7, img_fit, 0.3, 0)

        info = self.ft.cal_rad_and_cendev(self.ave_fit_l, self.ave_fit_r)
        radius_of_curvature = np.min(info[0])
        self.n_rads.append(radius_of_curvature)
        self.radius_of_curvature = np.mean(self.n_rads)
        center_deviation = info[1]
        self.n_cendevs.append(center_deviation)
        self.center_deviation = np.mean(self.n_cendevs)

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

    def sanity_check_fit(self):
        '''
        Checking that they have similar curvature,
        separated by approximately the right distance horizontally
        and roughly parallel.
        '''
        # check fit params

        xl, yl, xr, yr = self.line_points
        if len(xl) > 0:
            fit_l = np.polyfit(yl, xl, 2)
        else:
            fit_l = self.ave_fit_l
        if len(xr) > 0:
            fit_r = np.polyfit(yr, xr, 2)
        else:
            fit_r = self.ave_fit_r

        if len(self.n_fits_l) == 0: self.n_fits_l.append(fit_l)
        if len(self.n_fits_r) == 0: self.n_fits_r.append(fit_r)

        good_a_diffs = np.abs(fit_l[0] - fit_r[0]) < 4e-4
        good_b_diffs = np.abs(fit_l[1] - fit_r[1]) < 0.4
        good_pos = (-100 < fit_l[2] < 740) and (540 < fit_r[2] < 1380)
        good_c_diffs = 500 < (fit_r[2] - fit_l[2]) < 1000

        if good_a_diffs and good_b_diffs and good_pos and good_c_diffs:
            self.good_fit = True
            self.use_fit_margin = True
            self.n_fits_l.append(fit_l)
            self.n_fits_r.append(fit_r)
        else:
            self.good_fit = False
        self.n_good_fit.append(self.good_fit)
        if np.mean(self.n_good_fit) <= 0.3:
            self.use_fit_margin = False

        self.ave_fit_l = np.mean(np.array(self.n_fits_l), axis=0)
        self.ave_fit_r = np.mean(np.array(self.n_fits_r), axis=0)


if __name__ == '__main__':
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
    linefinder = LaneLineFinder4Video(undist=undistort, linefilter=lf, warper=wp, fittool=ft)

    video_out = '../project_video_out.mp4'
    clip = VideoFileClip('../project_video.mp4')
    white_clip = clip.fl_image(linefinder.apply)
    white_clip.write_videofile(video_out, audio=False)

    img_size = (1280, 720)
    src = np.float32(
        [[(img_size[0] / 2) - 75, img_size[1] / 2 + 120],
         [0, img_size[1]],
         [img_size[0], img_size[1]],
         [(img_size[0] / 2) + 100, img_size[1] / 2 + 120]])
    dst = np.float32(
        [[img_size[0] / 8, 0],
         [img_size[0] / 8, img_size[1]],
         [img_size[0] * 7/8, img_size[1]],
         [img_size[0] * 7/8, 0]])

    wp = WarpPerspective(src=src, dst=dst)
    ft = LaneLineFitTool(ym_per_pix=28/720)
    linefinder = LaneLineFinder4Video(undist=undistort, linefilter=lf, warper=wp, fittool=ft)

    video_out = '../challenge_video_out.mp4'
    clip = VideoFileClip('../challenge_video.mp4')
    white_clip = clip.fl_image(linefinder.apply) #NOTE: this function expects color images!!
    white_clip.write_videofile(video_out, audio=False)