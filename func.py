import cv2
import numpy as np

# variables
FRAME_ADJUST_BLUR_LEVEL = 3
THRESHOLD_MIN_AREA = 100
COLOR_GREEN = (165, 255, 59)
COLOR_PINK = (165, 59, 255)
COLOR_WHITE = (255, 255, 255)
LINE_WIDTH = 2
WINDOW_NAME_1 = 'Original'
WINDOW_NAME_2 = 'Adjusted'


def show_cam(device: int = 0, width: int = 720):
    """
    shows webcam
    """

    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, (9/16 * width))

    while True:
        _, frame = cap.read()
        adjusted_frame = frame_adjustment(frame=frame,
                                          threshold=True,
                                          value1=75,
                                          value2=150)

        # find playingcards
        end_frame = find_playingcards(adjusted_frame=adjusted_frame,
                                      display_frame=frame,
                                      card_value_1=0,
                                      card_value_2=150
                                      )

        # show image/ image-stream and break
        cv2.imshow(WINDOW_NAME_1, end_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # clean up
    cap.release()
    cv2.destroyAllWindows()


def show_image(image: str):
    """
    shows image
    """

    frame = cv2.imread(image, cv2.CAP_DSHOW)

    # resize to fit screen
    factor = 600
    height, width, _ = frame.shape
    if height > factor or width > factor:
        tmp_lst = list([int(height / factor), int(width / factor)])
        frame = cv2.resize(frame, dsize=(int(width/max(tmp_lst)), int(height/max(tmp_lst))))

    # adjust frame
    adjusted_frame = frame_adjustment(frame=frame,
                                      gray=True,
                                      threshold=True,
                                      value1=100,
                                      value2=255)

    # find playingcards
    end_frame = find_playingcards(adjusted_frame=adjusted_frame,
                                  display_frame=None,
                                  card_value_1=75,
                                  card_value_2=150
                                  )

    # show image/ image-stream and break
    cv2.imshow(WINDOW_NAME_1, end_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def frame_adjustment(frame,
                     blur=True,
                     gray=True,
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
                              type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if canny:
        frame = cv2.Canny(frame, value1, value2)
    if flip:
        frame = cv2.flip(frame, 1)
    return frame


def find_playingcards(adjusted_frame,
                      display_frame=None,
                      card_value_1=20,
                      card_value_2=1):

    contours, hierarchy = cv2.findContours(adjusted_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_is_card = np.zeros(len(contours), dtype=int)

    for i, contour in enumerate(contours):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

        if (hierarchy[0][i][3] == -1) and (len(approx) == 4):  # Find 'parent' object with 4 sides
            if display_frame is None:  # use adjusted frame to add color contours
                display_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_GRAY2RGB)
            # 1. draw green contour line
            cv2.drawContours(display_frame,
                             [contour],
                             0,
                             COLOR_GREEN,
                             LINE_WIDTH)

            # 2. extract card
            # Find width and height of card's bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            card = flattener(display_frame, approx, w, h)
            cv2.imshow('isolated card', card)
            print(w, h)
            # 3. show corner
            corner = card[0:int(h/7.5), 0:int(w/8)]
            corner_zoom = cv2.resize(corner, (0, 0), fx=4, fy=4)
            cv2.imshow('Corner', corner_zoom)
# https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector/blob/master/Cards.py
    cv2.line(display_frame, (133, 94), (526, 598), COLOR_PINK, LINE_WIDTH)
    cv2.line(display_frame, (433, 96), (134, 161), COLOR_PINK, LINE_WIDTH)
    cv2.line(display_frame, (134, 161), (222, 596), COLOR_PINK, LINE_WIDTH)
    cv2.line(display_frame, (222, 596), (523,530), COLOR_PINK, LINE_WIDTH)
    cv2.line(display_frame, (523,530), (433,96), COLOR_PINK, LINE_WIDTH)
    return display_frame


def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4, 2), dtype="float32")

    s = np.sum(pts, axis=2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8 * h:  # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2 * h:  # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.

    if 0.8 * h < w < 1.2 * h:  # If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0]  # Top left
            temp_rect[1] = pts[0][0]  # Top right
            temp_rect[2] = pts[3][0]  # Bottom right
            temp_rect[3] = pts[2][0]  # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0]  # Top left
            temp_rect[1] = pts[3][0]  # Top right
            temp_rect[2] = pts[2][0]  # Bottom right
            temp_rect[3] = pts[1][0]  # Bottom left

    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

    return warp

