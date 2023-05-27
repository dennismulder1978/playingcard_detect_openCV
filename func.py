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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        frame = cv2.threshold(frame, 110, 255, cv2.THRESH_BINARY)[1]

        contours = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        print (contours)
        # for contour in contours:
        #     M = cv2.moments(contour)
        #     if M['m00'] != 0:
        #         cX = int(M['m10'] / M['m00'])
        #         cY = int(M['m01'] / M['m00'])
        #         shape = card_detector(contour)
        #         cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def card_detector(contour):
    shape = ""
    contour_perimeter = 0.01 * cv2.arcLength(contour, True)
    approx_poly_curve = cv2.approxPolyDP(contour, contour_perimeter, True)

    if len(approx_poly_curve) == 4:
        # print(f"Length: {len(approx_poly_curve)} -- {cv2.boundingRect(approx_poly_curve)}")
        (x, y, w, h) = cv2.boundingRect(approx_poly_curve)
        shape = 'quad'

    return shape

