#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:59:58 2017

@author: alpha
"""

import pickle
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

class CameraCalibrator(object):

    def __init__(self):
        # Declare calibration matrixes.
        self.mtx = None
        self.dist = None

    def _gen_points(self, patten):
        '''Generate object points of one image using given patten.'''
        points = np.zeros((np.multiply(*patten), 3), np.float32)
        points[:, :2] = np.mgrid[0:patten[0], 0:patten[1]].T.reshape(-1, 2)
        return points

    def calibrate(self, chessboard_img_path, chessboard_patten=(9, 6), img_size=(1280, 720)):
        # Search pattens that may be smaller than default
        nx, ny = chessboard_patten
        pattens = [(x, y) for x in range(nx, 2, -1) for y in range(ny, 2, -1)]
        # calibrate using all images.
        images = glob.glob(os.path.join(chessboard_img_path, '*.jpg'))
        objpoints = []  # 3D object points of the chessboard.
        imgpoints = []  # 2D corner points in image plane.
        for idx, img_path in enumerate(images):
            img = plt.imread(img_path)
            if (img.shape[1], img.shape[0]) != img_size:
                img = cv2.resize(img, img_size)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            for patten in pattens:
                ret, corners = cv2.findChessboardCorners(gray, patten, None)
                if ret:
                    points = self._gen_points(patten)
                    objpoints.append(points)
                    imgpoints.append(corners)
                    break
                else:
                    continue
        assert len(imgpoints) > 0, 'No corners found!'
        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints,
                                                           img_size, None, None)
        print('Calibration Done!')

    def undistort(self, img):
        assert self.mtx is not None, 'Er... No calibration matrix found!'
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def save_matrix(self, fname='camera_mtx_dist.pk'):
        camera_mtx_dist = {}
        camera_mtx_dist['mtx'] = self.mtx
        camera_mtx_dist['dist'] = self.dist
        with open(fname, 'wb') as f:
            pickle.dump(camera_mtx_dist, f)
        print('Camera calibration mtx and dist saved to {}'.format(fname))

    def load_matrix(self, fname):
        with open(fname, 'rb') as f:
            camera_mtx_dist = pickle.load(f)
        self.mtx = camera_mtx_dist['mtx']
        self.dist = camera_mtx_dist['dist']
        print('Camera calibration mtx and dist loaded from {}'.format(fname))


if __name__ == '__main__':
    cc = CameraCalibrator()
    cc.calibrate(chessboard_img_path='../camera_cal/')
    cc.save_matrix(fname='../camera_mtx_dist.pk')
    undistort = cc.undistort  # rename function for easy use.
    img = plt.imread('../camera_cal/calibration1.jpg')
    undistorted = undistort(img)
    plt.figure(figsize=(12, 4), dpi=100)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(undistorted)
    plt.title('Undistorted Image')
    plt.savefig('../chessboard_undistortion.jpg')

    test_img_paths = sorted(glob.glob('../test_images/*.jpg'))
    for test_img_path in test_img_paths:
        img = plt.imread(test_img_path)
        undist = undistort(img)
        basename = os.path.basename(test_img_path)
        plt.imsave(os.path.join('../output_images',
                                basename.split('.')[0]+'_undist.jpg'),
                   undist, format='jpg')

    img = plt.imread('../test_images/test5.jpg')
    img_undist = plt.imread('../output_images/test5_undist.jpg')
    plt.figure(figsize=(12, 4), dpi=100)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(img_undist)
    plt.title('Undistorted Image')
    plt.savefig('../undistortion_example.jpg')