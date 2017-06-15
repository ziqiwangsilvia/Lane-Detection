# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 20:15:54 2017

@author: WangZ
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Color(I):

    r, c, channel = I.shape
    I2 = np.zeros((r,c,channel), np.uint8)

    # convert from RGB to HSV
    I_HSV = cv2.cvtColor(I,cv2.COLOR_RGB2HSV)
    h = I_HSV[:,:,0]
    s = I_HSV[:,:,1]
    v = I_HSV[:,:,2]

    # set the up and bottom threshold for white and yellow
#==============================================================================
#     low = np.array([30,0,150])
#     high = np.array([120,255,255])
#   
#==============================================================================
    low = np.array([20, 50, 150])
    high = np.array([100, 255, 255])
    # segment ROI with respect to color range
    I2 = 255*((h>low[0]) & (h<high[0]) & (s>low[1]) & (s<high[1]) & (v>low[2]) & (v<high[2])).astype('uint8')
    plt.figure(1)
    plt.imshow(I2, cmap = 'gray')
    return I2