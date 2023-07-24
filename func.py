import cv2

# variables
FRAME_ADJUST_BLUR_LEVEL = 3
THRESHOLD_MIN_AREA = 100
COLOR_GREEN = (165, 255, 59)
COLOR_PINK = (165, 59, 255)
COLOR_WHITE = (255, 255, 255)
LINE_WIDTH = 3
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
                                      original_frame=frame,
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
                                      value1=75,
                                      value2=150)

    # find playingcards
    end_frame = find_playingcards(adjusted_frame=adjusted_frame,
                                  original_frame=frame,
                                  card_value_1=0,
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
                      original_frame=None,
                      card_value_1=400,
                      card_value_2=1):
    if original_frame is None:
        original_frame = adjusted_frame

    contours = cv2.findContours(adjusted_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > card_value_1:
            try:
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2RGB)
            except:
                pass
            cv2.drawContours(original_frame,
                             [contour],
                             0,
                             COLOR_GREEN,
                             LINE_WIDTH)

    return original_frame

