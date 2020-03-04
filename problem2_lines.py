# Project 2 for 673

import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
import helpers

#get the image
img =[cv.imread(File) for File in glob.glob("./Videos/data_1/data/*.png")]
img2 = np.array(img)

#num = int(input("Enter the number of the picture from 0-302: ")) # uncomment this line to use a different image
num=230 # best image
while True:
    #show the original image
    file = img2[num]
    # 4 points on the original image's lane lines (two from each line)
    source = np.array([[0,432], [280,350], [763,350], [820,432]])
    # 4 new points line) for where the image should line up
    dest = np.array([[294.0, 463.0], [294.0, 341.0], [892.0, 341.0], [892.0, 463.0]])
    cv.line(file, (0,432), (280,350), (0,255,0), 2)
    cv.line(file, (763,350), (820,432), (0, 255, 0), 2)
    cv.imshow("rezied src", file)

    # find homography H from the given 8 points
    h, status = cv.findHomography(source, dest)
    # create the top down view
    #newimg = cv.resize(file,(1000,500))
    unwarped = cv.warpPerspective(file, h, (1000, 500))
    cv.imshow("unwarped", unwarped)

    # threshold the image in binary
    unwarped_gray = cv.cvtColor(unwarped, cv.COLOR_BGR2GRAY)
    ret, BW_lanes = cv.threshold(unwarped_gray, 200, 255, cv.THRESH_BINARY)
    cv.imshow("BW Lanes", BW_lanes)
    # perform Canny edge detection
    edges = cv.Canny(BW_lanes, 50, 100, apertureSize=3)
    laplacian = cv.Laplacian(BW_lanes, cv.CV_64F)
    sobelx = cv.Sobel(BW_lanes, cv.CV_64F, 1, 0, ksize=5)  # x
    sobely = cv.Sobel(BW_lanes, cv.CV_64F, 0, 1, ksize=5)  # y
    cv.imshow("Canny", edges)
    cv.imshow("laplacian", laplacian)
    cv.imshow("Sobelx", sobelx)
    cv.imshow("Sobely", sobely)
    # quit program if user presses escape or 'q'

    key = cv.waitKey() & 0xFF
    if key == 27 or key == ord("q"):
        break

    

cv.destroyAllWindows()

