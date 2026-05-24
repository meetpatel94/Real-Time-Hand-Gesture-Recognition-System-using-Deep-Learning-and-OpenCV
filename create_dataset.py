import cv2
import numpy as np
import os
import time

# ==========================
# USER INPUT
# ==========================

gesture_name = input(
    "Enter gesture name (A-Z / SPACE / DELETE / CLEAR): "
).upper()

mode = "train"

save_path = f"dataset/{mode}/{gesture_name}"

if not os.path.exists(save_path):
    os.makedirs(save_path)

TOTAL_IMAGES = 500
CAPTURE_DELAY = 0.1

# ==========================
# ROI
# ==========================

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

last_capture = time.time()


# ==========================
# HAND SEGMENTATION
# ==========================

def segment_hand(frame):

    # OTSU Threshold
    _, thresholded = cv2.threshold(
        frame,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5,5), np.uint8)

    thresholded = cv2.morphologyEx(
        thresholded,
        cv2.MORPH_OPEN,
        kernel
    )

    thresholded = cv2.morphologyEx(
        thresholded,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        thresholded.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    hand_segment = max(
        contours,
        key=cv2.contourArea
    )

    if cv2.contourArea(hand_segment) < 3000:
        return None

    return thresholded, hand_segment


# ==========================
# CAMERA
# ==========================

cam = cv2.VideoCapture(0)

cv2.namedWindow(
    "Dataset Collection",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "Thresholded",
    cv2.WINDOW_NORMAL
)

num_imgs_taken = 0

print("\nMove hand slowly")
print("Left Right Up Down")
print("ESC → Exit\n")


while True:

    ret, frame = cam.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    frame_copy = frame.copy()

    roi = frame[
        ROI_top:ROI_bottom,
        ROI_right:ROI_left
    ]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (7,7),
        0
    )

    hand = segment_hand(gray)

    if hand is not None:

        thresholded, hand_segment = hand

        cv2.drawContours(
            frame_copy,
            [hand_segment + (ROI_right, ROI_top)],
            -1,
            (255,0,0),
            2
        )

        cv2.imshow(
            "Thresholded",
            thresholded
        )

        current_time = time.time()

        # Save image with delay
        if current_time-last_capture > CAPTURE_DELAY:

            if num_imgs_taken < TOTAL_IMAGES:

                resized = cv2.resize(
                    thresholded,
                    (64,64)
                )

                file_name = os.path.join(
                    save_path,
                    f"{num_imgs_taken}.jpg"
                )

                cv2.imwrite(
                    file_name,
                    resized
                )

                num_imgs_taken += 1

                last_capture = current_time

        cv2.putText(
            frame_copy,
            f"{gesture_name}: {num_imgs_taken}/{TOTAL_IMAGES}",
            (150,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        if num_imgs_taken >= TOTAL_IMAGES:

            print(
                f"\nCompleted {gesture_name}"
            )

            break

    cv2.rectangle(
        frame_copy,
        (ROI_left,ROI_top),
        (ROI_right,ROI_bottom),
        (255,128,0),
        2
    )

    cv2.imshow(
        "Dataset Collection",
        frame_copy
    )

    key = cv2.waitKey(1)

    if key == 27:
        break


cam.release()
cv2.destroyAllWindows()