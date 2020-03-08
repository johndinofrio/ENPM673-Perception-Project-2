import numpy as np
import cv2 as cv
import helpers
from matplotlib import pyplot as plt

img1 = cv.imread('./testimg1.png',0)
img2 = cv.imread('./testimg2.png',0)
img3 = cv.imread('./testimg3.png',0)

img1 = cv.equalizeHist(img1)
img2 = cv.equalizeHist(img2)
img3 = cv.equalizeHist(img3)


cv.imshow("img1", img1)
cv.imshow("img2", img2)
cv.imshow("img3", img3)

cv.waitKey(0)