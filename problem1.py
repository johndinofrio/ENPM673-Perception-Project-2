# Project 2 for 673

import numpy as np
import cv2 as cv
import helpers
from matplotlib import pyplot as plt

video=cv.VideoCapture('./Videos/NightDrive-2689.mp4')

# ask user to select playback style
print("How would you like video to play?")
playback = int(input("0 - hit a key to advance each frame\n1 - the next frame is automatically processed after the previous: "))
if playback not in [0,1]:
    print("Unknown value,", playback)
    exit(0)

frame = helpers.frame()
gamma = helpers.frame()
hist = helpers.frame()
while video.isOpened():
    # increment the frame number and print it
    print("Frame:",frame.increment_frame())

    # current frame is taken in, grabbed returns true or false if frame is grabbed
    (grabbed, frame.image) = video.read()
    # if we tried to grab a frame but the video ended, break the loop
    if grabbed == False:
        break

    frame.resize(0.3)
    cv.imshow("rezied src", frame.image)
    gamma.image = frame.image.copy()
    hist.image = frame.image.copy()


    gamma.image, table = gamma.adjust_gamma(2.0)
    # frame.smooth(3)

    cv.imshow("gamma",gamma.image)

    print(np.shape(hist.image))

    # histogram = np.histogram(hist.image.ravel(),256,[0,256])
    hist.image = cv.cvtColor(hist.image, cv.COLOR_RGB2GRAY)
    hist.image = cv.equalizeHist(hist.image)
    plt.hist(hist.image.ravel(),256,[0,256])
    # #print(histogram)
    plt.show()

    cv.imshow("histogram",hist.image)

    #hist.equalizeBins(4)
    #cv.imshow("binned hist", hist.image)

    # cum=[]
    # for i in histogram[1]:
    #     i = int(i)
    #     if i != 0 and i <=255:
    #         cum.append(cum[i-1]+histogram[0][i])
    #     elif i == 0:
    #         cum.append(histogram[0][i])
    #
    # hPrime = []
    # for i in range(0,256):
    #     port = cum[i]/cum[255]
    #     hPrime.append(int(port*255))
    # print(cum[255])
    # display = []
    # for r in range(256):
    #     display.append([histogram[0][r], r, table[r], hPrime[r]])
    #     print(display[r])



    key = cv.waitKey(playback) & 0xFF
    if key == 27 or key == ord("q"):
        break

video.release()
cv.destroyAllWindows()

