# Project 1 for 673 - perception for autonomous robots
# This program processes video files of different AR tags

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
while video.isOpened():
    # increment the frame number and print it
    print("Frame:",frame.increment_frame())

    # current frame is taken in, grabbed returns true or false if frame is grabbed
    (grabbed, frame.image) = video.read()
    # if we tried to grab a frame but the video ended, break the loop
    if grabbed == False:
        break

    frame.resize(0.5)

    cv.imshow("rezied src", frame.image)
    frame.image = frame.adjust_gamma(2.0)
    frame.smooth(3)
    cv.imshow("gamma",frame.image)

    key = cv.waitKey(playback) & 0xFF
    if key == 27 or key == ord("q"):
        break

video.release()
cv.destroyAllWindows()

