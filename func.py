import numpy as np
import cv2


def show_cam(height: int = 1024):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1.25 * height)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        _, frame = cap.read()
        # frame adjustment
        frame = frame_adjustment(frame=frame, blur=True, threshold=True)

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
                     threshold=False,
                     flip=False):
    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if threshold:
        if not gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(frame, thresh=125, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if blur:
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
    if canny:
        frame = cv2.Canny(frame, 30, 80, 8)
    if flip:
        frame = cv2.flip(frame, 1)
    return frame


def find_playingcards(frame):
    contours = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    threshold_min_area = 10

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > threshold_min_area:
            cv2.drawContours(frame, [contour], 0, (36, 255, 12), 3)
        contour_perimeter = 0.16 * cv2.arcLength(contour, True)
        approx_poly_curve = cv2.approxPolyDP(contour, contour_perimeter, True)
        if len(approx_poly_curve) == 4:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.putText(frame, f'{x}, {y}, {w}, {h}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)

    return frame

    # for contour in contours:
    #     M = cv2.moments(contour)
    #     if M['m00'] != 0:
    #         cX = int(M['m10'] / M['m00'])
    #         cY = int(M['m01'] / M['m00'])
    #         shape = ""
    #         contour_perimeter = 0.15 * cv2.arcLength(contour, True)
    #         approx_poly_curve = cv2.approxPolyDP(contour, contour_perimeter, True)
    #         if len(approx_poly_curve) == 4:
    #             (x, y, w, h) = cv2.boundingRect(approx_poly_curve)
    #             shape = 'quad'
    #         cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
