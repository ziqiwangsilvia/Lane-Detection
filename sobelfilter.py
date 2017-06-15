# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 20:09:45 2017

@author: WangZ
"""

import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def sobel_filter(I):
    r,c,channel = I.shape

    # convert RGB to HSV
    I_HSV = cv2.cvtColor(I,cv2.COLOR_RGB2HSV)
    H = I_HSV[:,:,0]
    #S = I_HSV[:,:,1]
    V = I_HSV[:,:,2]

    # Sobel kernel
    Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    # 2D convolutional
    h_convx = signal.convolve2d(H, Gx, mode = 'same') + 2e-16
    h_convy = signal.convolve2d(H, Gy, mode = 'same') + 2e-16
    #s_convx = signal.convolve2d(S, Gx, mode = 'same') + 2e-16
    #s_convy = signal.convolve2d(S, Gy, mode = 'same') + 2e-16
    v_convx = signal.convolve2d(V, Gx, mode = 'same') + 2e-16
    v_convy = signal.convolve2d(V, Gy, mode = 'same') + 2e-16

    # get magnitude of gradient at each pixel
    GH = np.sqrt(h_convx**2 + h_convy**2) + 2e-16
    #GS = np.sqrt(s_convx**2 + s_convy**2) + 2e-16
    GV = np.sqrt(v_convx**2 + v_convy**2) + 2e-16

    # get direction of gradient at each pixel
    direc_h = np.arctan(h_convy/h_convx)
    #direc_s = np.arctan(s_convy/s_convx)
    direc_v = np.arctan(v_convy/v_convx)

    threshold_low = 50
    threshold_high = 250
    direc_low = 0.2
    direc_high = 1.
#==============================================================================
#     #basic video
#     direc_low = 1
#     direc_high = 2.5
#==============================================================================


    SH = 255*((GH > threshold_low) & (GH < threshold_high) & (abs(direc_h) >= direc_low) & (abs(direc_h) <= direc_high))
    #SS = 255*((GS > threshold_low) & (GS < threshold_high) & (abs(direc_s) >= direc_low) & (abs(direc_s) <= direc_high))
    SV = 255*((GV > threshold_low) & (GV < threshold_high) & (abs(direc_v) >= direc_low) & (abs(direc_v) <= direc_high))
         

    SV_combine = cv2.bitwise_or(SH,SV)
    SV_combine = cv2.GaussianBlur(SV_combine.astype('uint8'),(5,5),0)
    plt.figure(2)
    plt.imshow(SV_combine,cmap='gray')      
    cv2.imwrite("fil8.jpg", SV_combine)       
    return SV_combine        