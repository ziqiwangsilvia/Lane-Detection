# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:07:16 2017

@author: WangZ
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Line():
    def __init__(self, img, prev_line = None):
        self.img = img;
        self.prev_line = prev_line
        self.right_fit = None
        self.left_fit = None
        if prev_line != None:
            self.lane_width = prev_line.lane_width
        else:
            self.lane_width = 760
        
    def histogram_find_lane(self, window_num=10, window_width=50):
        # count the lower half of the img
        img = self.img
        histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
        mid_point = histogram.shape[0]/2
        margin = 10
        left_peak = np.argmax(histogram[margin:mid_point-margin])
        right_peak = np.argmax(histogram[mid_point + margin:2 * mid_point - margin])+ mid_point
        window_height = img.shape[0]/ window_num 
        # store left and right points
        left_points = []
        right_points = []

        for num in range(window_num):
            window_lower = self.img.shape[0] - num * window_height
            window_upper = max(window_lower - window_height, 0)
            # for left line
            window_left_1 = left_peak - window_width
            window_right_1 = left_peak + window_width
            windowed_img_1 = img[window_upper:window_lower,window_left_1:window_right_1]
            # for right line
            window_left_2 = right_peak - window_width
            window_right_2 = right_peak + window_width    
            windowed_img_2 = img[window_upper:window_lower,window_left_2:window_right_2]
    
            nonzero_idx_1 = windowed_img_1.nonzero()
            left_point = [[window_upper + nonzero_idx_1[0][i],window_left_1 + nonzero_idx_1[1][i]] for  i in range(len(nonzero_idx_1[0]))]
            left_points += left_point
            nonzero_idx_2 = windowed_img_2.nonzero()
            right_point = [[window_upper + nonzero_idx_2[0][i],window_left_2 + nonzero_idx_2[1][i]] for  i in range(len(nonzero_idx_2[0]))]
            right_points += right_point
    
            # calibrate peak
            try: 
                left_peak = window_left_1 + int(np.median(nonzero_idx_1[1]))
                right_peak = window_left_2 + int(np.median(nonzero_idx_2[1]))
                #print left_peak, right_peak
            except:
                continue
                
        leftx = [p[1] for p in left_points]
        lefty = [p[0] for p in left_points]
        rightx = [p[1] for p in right_points]
        righty = [p[0] for p in right_points]
        
        print "histogram finding!"
        return leftx, lefty, rightx, righty
    
    def smart_find_lane(self):
        prev_left_fit = self.prev_line.left_fit
        prev_right_fit = self.prev_line.right_fit
        margin = 50
        yvals = np.linspace(0, 100, num=101)*7.2
        yvals = yvals[::-1]
        # store left and right point
        nonzero_idx = self.img.nonzero()
        nonzero_x = np.array(nonzero_idx[1])
        nonzero_y = np.array(nonzero_idx[0])
        
        left_idx = (
            ( nonzero_x > (
                prev_left_fit[0] * (nonzero_y ** 2) + prev_left_fit[1] * nonzero_y + prev_left_fit[2] - margin
                          )) & 
            (nonzero_x < (
                prev_left_fit[0] * (nonzero_y ** 2) + prev_left_fit[1] * nonzero_y + prev_left_fit[2] + margin))
                      )
        right_idx = (
            ( nonzero_x > (
                prev_right_fit[0] * (nonzero_y ** 2) + prev_right_fit[1] * nonzero_y + prev_right_fit[2] - margin
                          )) & 
            (nonzero_x < (
                prev_right_fit[0] * (nonzero_y ** 2) + prev_right_fit[1] * nonzero_y + prev_right_fit[2] + margin))
                      )
        
        leftx = list(nonzero_x[left_idx])
        lefty = list(nonzero_y[left_idx])
        rightx = list(nonzero_x[right_idx])
        righty = list(nonzero_y[right_idx])
        
        print "smart finding!"
        #print len(leftx), len(rightx)
        return leftx, lefty, rightx, righty
        
    
    def fit(self, leftx, lefty, rightx, righty, run_test = False, display=False):  
        
        if len(lefty)< 50:
            right_fit = np.polyfit(righty,rightx,2)
            left_fit = right_fit.copy()
            left_fit[2] -= self.lane_width
            #print "left is missing!"
        elif len(righty)< 50:
            left_fit = np.polyfit(lefty,leftx,2)
            right_fit = left_fit.copy()
            right_fit[2] += self.lane_width
            #print "right is missing!"
        else:
            left_fit = np.polyfit(lefty,leftx,2)
            right_fit = np.polyfit(righty,rightx,2)
        

        yvals = np.linspace(0, 100, num=101)*7.2
        yvals = yvals[::-1]

        left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
        right_fitx =right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]   
        
        if run_test:
            test_result = self.test_fit(right_fit, left_fit)
            if test_result == 'pass':
                pass
            elif test_result == 'right':
                right_fit = left_fit.copy()
                right_fit[2] += self.lane_width
                #print "right is bad!"
            elif test_result == 'left':
                left_fit = right_fit.copy()
                left_fit[2] -= self.lane_width  
                #print "left is bad!"
                
        self.right_fit = right_fit
        self.left_fit = left_fit
        self.lane_width = right_fit[2] - left_fit[2]
                
        if display:
            img_color = cv2.cvtColor(self.img,cv2.COLOR_GRAY2BGR)*255
            img_color[lefty,leftx] = [0,0,255]
            img_color[righty,rightx] = [0,255,0]
            plt.imshow(img_color)
            plt.plot(left_fitx, yvals, color='blue', linewidth=3)
            plt.plot(right_fitx, yvals, color='green', linewidth=3)
            plt.show()
        return left_fitx, right_fitx, yvals
    
    def find_lane_and_fit(self, prev_left_fitx, prev_right_fitx, prev_yvals, prev_left_fit, prev_right_fit, display = False):
        if self.prev_line == None:
            leftx, lefty, rightx, righty = self.histogram_find_lane()
            left_fitx, right_fitx, yvals = self.fit(leftx, lefty, rightx, righty, run_test=True, display=display)
        else:
            leftx, lefty, rightx, righty = self.smart_find_lane()
            if len(leftx) == 0 or len(rightx) == 0 or len(lefty) == 0 or len(righty) == 0:
                left_fitx, right_fitx, yvals = prev_left_fitx.copy(), prev_right_fitx.copy(), prev_yvals.copy()
                self.left_fit, self.right_fit = prev_left_fit.copy(), prev_right_fit.copy()
            else:
                left_fitx, right_fitx, yvals = self.fit(leftx, lefty, rightx, righty, run_test=True, display=display)
                
                if abs(self.lane_width - self.prev_line.lane_width) > 20 or  self.good_fit_quality()==False:
                    print "smart finding fiailed! Try histogram again!"
                    leftx, lefty, rightx, righty = self.histogram_find_lane()
                    if len(leftx) == 0 or len(rightx) == 0 or len(lefty) == 0 or len(righty) == 0:
                        left_fitx, right_fitx, yvals = prev_left_fitx.copy(), prev_right_fitx.copy(), prev_yvals.copy()
                        self.left_fit, self.right_fit = prev_left_fit.copy(), prev_right_fit.copy()
                    else:
                        left_fitx, right_fitx, yvals = self.fit(leftx, lefty, rightx, righty, run_test=True, display=display)
                        if self.good_fit_quality() == False:
                            left_fitx, right_fitx, yvals = prev_left_fitx.copy(), prev_right_fitx.copy(), prev_yvals.copy()   
                            self.left_fit, self.right_fit = prev_left_fit.copy(), prev_right_fit.copy()
        return left_fitx, right_fitx, yvals, self.left_fit, self.right_fit
            
    def good_fit_quality(self, threshold = 1e-6):
        fit_diff = sum(np.square(self.right_fit[0:-1] - self.left_fit[0:-1]))
        fit_norm = abs(self.right_fit[0]) + abs(self.left_fit[0])
        if fit_diff > threshold or fit_norm > threshold:
            #print fit_diff,fit_norm
            return False
        else:
            return True
    
    def test_fit(self,right_fit, left_fit,threshold = 1e-4):
        right_coef = abs(right_fit[0])
        left_coef = abs(left_fit[0])
        if right_coef > threshold or  left_coef > threshold:
            if left_coef > right_coef:
                return 'left'
            else:
                return 'right'
        else:
            return 'pass' 
