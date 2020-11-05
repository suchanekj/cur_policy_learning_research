# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
import shutil
import os
import pyqrcodeng as pyqrcode

import frame_loader
import mine_detection

record = None
source = "maze.avi"
# source = "table2.avi"
# source = 1

if type(source) == int:
    fl = frame_loader.CameraFrameLoader(source, record)
else:
    fl = frame_loader.VideoFrameLoader(source, record)

md = mine_detection.MineDetection()

# keep looping
while True:
    frame = fl.get_frame_cropped()
    mask = md.get_mines_mask(frame)
    frame = md.get_mine_positions(frame)

    # show the frame to our screen
    display = np.concatenate((frame, np.stack((mask, mask, mask), 2)), 1)
    cv2.imshow("Frame", display)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# close all windows
cv2.destroyAllWindows()
fl.writer.release()
del fl
del md
