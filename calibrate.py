# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:43:56 2017

@author: WangZ
"""

import cv2

def Calibrate(frame, camera_mtx, dist_coef):
    

    image_ud = cv2.undistort(frame, camera_mtx, dist_coef, None, camera_mtx)

    return image_ud

