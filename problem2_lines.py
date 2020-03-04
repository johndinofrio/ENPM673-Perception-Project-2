# Project 2 for 673

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import helpers

#get the image
img = cv.imread('./Videos/data_1/0000000000.png')


while (1):
    #show the original image
    cv.imshow("rezied src", img)

    #4 points on the original image's lane lines (two from each line) 
    source = np.array([[294.0,463.0],[510.0,331.0],[770.0,327.0],[892.0,472.0]])
    #4 new points line) for where the image should line up
    dest = np.array([[294.0,463.0],[294.0,341.0],[892.0,341.0],[892.0,463.0]])
    #find homography H from the given 8 points 
    h,status = cv.findHomography(source,dest)
    #create the top down view
    unwarped = cv.warpPerspective(img, h, (1000,500))
    cv.imshow("unwarped", unwarped)

    #threshold the image in binary
    unwarped_gray = cv.cvtColor(unwarped,cv.COLOR_BGR2GRAY)
    ret,BW_lanes = cv.threshold(unwarped_gray,200,255,cv.THRESH_BINARY)
    cv.imshow("BW Lanes", BW_lanes)
    #perform Canny edge detection
    edges = cv.Canny(BW_lanes,50,100,apertureSize = 3)
    
    #quit program if user presses escape or 'q'
    key = cv.waitKey() & 0xFF
    if key == 27 or key == ord("q"):
        break
    

cv.destroyAllWindows()

