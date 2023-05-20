import numpy as np
import cv2
from keras.models import load_model


def show_cam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bw_frame = cv2.threshold(gray_frame, 100, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(bw_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow('Frame', bw_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



