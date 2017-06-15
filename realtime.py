# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:52:32 2017

@author: WangZ
"""

import cv2
import numpy as np
from calibrate import Calibrate
from color import Color
from sobelfilter import sobel_filter
from perspective import Transform
from fitline import Fit

camera_mtx = np.array([[  1.15475333e+03,   0.00000000e+00,   6.71943091e+02],
                           [  0.00000000e+00,   1.14897531e+03,   3.85100896e+02],
                           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

dist_coef = np.array([ -2.49827121e-01,   8.22238034e-03,  -1.06658271e-03,
                          -4.21417236e-05,  -8.86958217e-02])
src = np.float32(
            [[323, 682],
             [1105, 682],
             [564, 483],
             [756, 483]])

dst = np.float32(
            [[300,720],
             [1000,720],
             [300,0],
             [1000,0]])

# read in frame and show
videoCapture = cv2.VideoCapture('original.mp4')
frames = []
i = 0
while True:
    success,frame = videoCapture.read()
    if success:
        cv2.imwrite("test_images/frame1/%d.png" %i,frame)
        frames.append(frame)
        i += 1
        
        ud = Calibrate(frame, camera_mtx, dist_coef)
        color = Color(ud)
        sobel = sobel_filter(ud)

        combine = cv2.bitwise_or(color,sobel)
        cs_combine = cv2.GaussianBlur(combine.astype('uint8'),(5,5),0)

        trans = Transform(cs_combine, dst, src)
        result = Fit(trans, dst, src, ud)
        #name = 'result' + str(i) + '.png'
        resultbgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imshow('Realtime', resultbgr)
        cv2.waitKey(1)
        #cv2.imwrite(name, resultbgr)
    else:
        break



