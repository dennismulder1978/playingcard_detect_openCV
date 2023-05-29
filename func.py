import cv2

# variables
FRAME_ADJUST_BLUR_LEVEL = 5
THRESHOLD_MIN_AREA = 100
COLOR_GREEN = (165, 255, 59)
COLOR_WHITE = (255, 255, 255)
LINE_WIDTH = 1
WINDOW_NAME_1 = 'Original'
WINDOW_NAME_2 = 'Adjusted'



def show_cam(device: int = 0, width: int = 720):
    """
    shows webcam
    """

    def nothing(x):
        pass

    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, (9/16 * width))
    cv2.namedWindow(WINDOW_NAME_1)
    cv2.namedWindow(WINDOW_NAME_2)
    cv2.createTrackbar('Threshold or canny', WINDOW_NAME_1, 0, 1, nothing)
    cv2.createTrackbar('Window value 1', WINDOW_NAME_1, 0, 255, nothing)
    cv2.createTrackbar('Window value 2', WINDOW_NAME_1, 0, 255, nothing)
    cv2.createTrackbar('Card value 1', WINDOW_NAME_2, 1, 255, nothing)
    cv2.createTrackbar('Card value 2', WINDOW_NAME_2, 0, 100, nothing)

    while True:
        _, frame = cap.read()
        # Trackbar values
        adjust_choice = cv2.getTrackbarPos('Threshold or canny', WINDOW_NAME_1)
        value1 = cv2.getTrackbarPos('Window value 1', WINDOW_NAME_1)
        value2 = cv2.getTrackbarPos('Window value 2', WINDOW_NAME_1)
        card_value1 = cv2.getTrackbarPos('Card value 1', WINDOW_NAME_2)
        card_value2 = cv2.getTrackbarPos('Card value 2', WINDOW_NAME_2)/100

        # frame adjustment
        if adjust_choice == 0:
            adjusted_frame = frame_adjustment(frame=frame, threshold=True, value1=value1)
        elif adjust_choice == 1:
            adjusted_frame = frame_adjustment(frame=frame, canny=True, value1=value1, value2=value2)

        # find playingcards
        end_frame = find_playingcards(adjusted_frame=adjusted_frame,
                                      original_frame=frame,
                                      card_value_1=card_value1,
                                      card_value_2=card_value2
                                      )

        # show image/ image-stream and break
        cv2.imshow('Adjusted', cv2.flip(adjusted_frame, 1))
        cv2.imshow(WINDOW_NAME_1, end_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # clean up
    cap.release()
    cv2.destroyAllWindows()


def frame_adjustment(frame,
                     blur=True,
                     gray=False,
                     canny=False,
                     threshold=False,
                     flip=False,
                     value1: int = 0,
                     value2: int = 0):
    if blur:
        frame = cv2.GaussianBlur(frame, (FRAME_ADJUST_BLUR_LEVEL, FRAME_ADJUST_BLUR_LEVEL), 0)
    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if threshold:
        if not gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.threshold(frame,
                              thresh=value1,
                              maxval=255,
                              type=cv2.THRESH_BINARY)[1]
    if canny:
        frame = cv2.Canny(frame, value1, value2)
    if flip:
        frame = cv2.flip(frame, 1)
    return frame


def find_playingcards(adjusted_frame,
                      original_frame=None,
                      card_value_1=1,
                      card_value_2=1):
    if original_frame is None:
        original_frame = adjusted_frame

    contours = cv2.findContours(adjusted_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > card_value_1:
            cv2.drawContours(original_frame,
                             [contour],
                             0,
                             COLOR_GREEN,
                             LINE_WIDTH)

        contour_perimeter = card_value_2 * cv2.arcLength(contour, True)
        approx_poly_curve = cv2.approxPolyDP(contour, contour_perimeter, True)
        if len(approx_poly_curve) == 4:

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.putText(original_frame,
                        f'{x}, {y}, {w}, {h}',
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        COLOR_WHITE,
                        LINE_WIDTH)

    return original_frame

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
