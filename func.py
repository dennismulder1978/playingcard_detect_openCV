import numpy as np
import cv2
import imutils


def show_cam(height: int = 1024):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1.25 * height)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        _, frame = cap.read()
        # frame adjustment
        frame = frame_adjustment(frame=frame, blur=True, canny=True)

        # find playingcards
        frame = find_playingcards(frame)

        # show image/ image-stream and break
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    # clean up
    cap.release()
    cv2.destroyAllWindows()


def frame_adjustment(frame,
                     gray=False,
                     blur=False,
                     canny=False,
                     flip=False):

    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur:
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
    if canny:
        frame = cv2.Canny(frame, 30, 80, 8)
    if flip:
        frame = cv2.flip(frame, 1)
    return frame


def find_playingcards(frame):
    contours = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            shape = ""
            contour_perimeter = 0.15 * cv2.arcLength(contour, True)
            approx_poly_curve = cv2.approxPolyDP(contour, contour_perimeter, True)
            if len(approx_poly_curve) == 4:
                (x, y, w, h) = cv2.boundingRect(approx_poly_curve)
                shape = 'quad'
            cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame



