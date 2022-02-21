import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


width_cam, height_cam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)
previous_time = 0

detector = htm.HandDetector(min_detection_confidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volume_range = volume.GetVolumeRange()
min_vol = volume_range[0]
max_vol = volume_range[1]
vol_bar = 400
vol_percent = 0

while True:
    success, img = cap.read()
    img = detector.find_hands(img, draw=False)
    lm_list = detector.find_position(img, draw=False)

    if len(lm_list) > 0:
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        vol = np.interp(length, [30, 250], [min_vol, max_vol])
        vol_bar = np.interp(length, [30, 250], [400, 150])
        vol_percent = np.interp(length, [30, 250], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)

        if length < 45:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f"Volume: {int(vol_percent)}%", (40, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
