# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 10:58:23 2017

@author: WangZ
"""

import cv2
import matplotlib.pyplot as plt

def Transform(image, dst, src):    
    

    imshape = image.shape

#==============================================================================
#     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# 
#     ax1.imshow(image, cmap="gray")
#     ax1.set_title('Undistorted Image',fontsize=20)
#==============================================================================

    M = cv2.getPerspectiveTransform(src, dst)
    #gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    warped = cv2.warpPerspective(image, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
#==============================================================================
#     ax2.imshow(warped, cmap="gray")
#     ax2.set_title('Perspective Transform',fontsize=20)
#==============================================================================
    return warped