# Project 2 for 673

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import helpers

video=cv.VideoCapture('./Videos/NightDrive-2689.mp4')

# ask user to select playback style
print("How would you like video to play?")
playback = int(input("0 - hit a key to advance each frame\n1 - the next frame is automatically processed after the previous: "))
if playback not in [0,1]:
    print("Unknown value,", playback)
    exit(0)

frame = helpers.frame()
lanes = helpers.frame()
#Prep: manually calculate homography function from one frame where road is straight. Camera is stationary in car,
# and so perspective is always the same.
#h=[[1,2,3],[4,5,6],[7,8,9]]
while video.isOpened():
    # increment the frame number and print it
    print("Frame:",frame.increment_frame())
    lanes.increment_frame()

    # current frame is taken in, grabbed returns true or false if frame is grabbed
    (grabbed, frame.image) = video.read()
    # if we tried to grab a frame but the video ended, break the loop
    if grabbed == False:
        break

    #resize image and show it as the source
    frame.resize(0.5)
    cv.imshow("rezied src", frame.image)

    #create a copy of the image frame against we we can find our lanes while preserving the main image
    lanes.image = frame.image
    #1. undistort frame using homography
    #frame.road = cv.warpPerspective(frame.image,h,etc,etc)
    #2. Denoise image
    #frame.road = cv2.GaussianBlur(self.image,(5,5),0)
    #3. Detect edges (Canny?)
    #4. Extract ROI (cut out sky)
    #frame.image = frame.image[:,250:500]

    #Detect Lanes
    #avg the intensity of each column of the image. We need to figure out what baseline asphalt would score as
    #anything higher is a possible lane marker. Draw a line there. Would work for straight lanes

    #Otherwise consider Hough lines, but they warn it also might not work well for curved roads. Can be done without
    #doing homogrpahy

    #Fit polynomial (ransac maybe)

    #calculate radius and distance from center (center pixel vs center line of lane using camera parameters to understand
    #the true distance
    #Project lane back into image

    key = cv.waitKey(playback) & 0xFF
    if key == 27 or key == ord("q"):
        break

video.release()
cv.destroyAllWindows()

