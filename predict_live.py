import cv2
import numpy as np
from tensorflow import keras
import os

# ======================
# LOAD MODEL
# ======================

if not os.path.exists("best_model.h5"):
    print("Model not found")
    exit()

model = keras.models.load_model("best_model.h5")
print("Model Loaded Successfully")

# ======================
# LABELS
# ======================

labels = {
0:'A',
1:'B',
2:'C',
3:'CLEAR',
4:'D',
5:'DELETE',
6:'E',
7:'F',
8:'G',
9:'H',
10:'I',
11:'J',
12:'K',
13:'L',
14:'M',
15:'N',
16:'O',
17:'P',
18:'Q',
19:'R',
20:'S',
21:'SPACE',
22:'T',
23:'U',
24:'V',
25:'W',
26:'X',
27:'Y',
28:'Z'
}

# ROI
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

sentence = ""
current_prediction = ""

# ======================
# HAND SEGMENT
# ======================

def segment_hand(frame):

    _, thresh = cv2.threshold(
        frame,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel=np.ones((5,5),np.uint8)

    thresh=cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh=cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours,_=cv2.findContours(
        thresh.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours)==0:
        return None

    hand=max(
        contours,
        key=cv2.contourArea
    )

    if cv2.contourArea(hand)<5000:
        return None

    return thresh,hand


# ======================
# CAMERA START
# ======================

cam=cv2.VideoCapture(0)

while True:

    ret,frame=cam.read()

    if not ret:
        break

    frame=cv2.flip(frame,1)

    frame_copy=frame.copy()

    roi=frame[
        ROI_top:ROI_bottom,
        ROI_right:ROI_left
    ]

    gray=cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray=cv2.GaussianBlur(
        gray,
        (7,7),
        0
    )

    hand=segment_hand(gray)

    if hand is not None:

        thresholded,hand_segment=hand

        cv2.drawContours(
            frame_copy,
            [hand_segment+(ROI_right,ROI_top)],
            -1,
            (255,0,0),
            2
        )

        cv2.imshow(
            "Threshold",
            thresholded
        )

        img=cv2.resize(
            thresholded,
            (64,64)
        )

        img=cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB
        )

        img=img/255.0

        img=np.reshape(
            img,
            (1,64,64,3)
        )

        prediction=model.predict(
            img,
            verbose=0
        )

        index=np.argmax(prediction)
        confidence=np.max(prediction)

        if confidence>0.90:
            current_prediction=labels[index]

    # Show prediction
    cv2.putText(
        frame_copy,
        "Current: "+current_prediction,
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    cv2.putText(
        frame_copy,
        "TEXT: "+sentence,
        (20,450),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        3
    )

    cv2.rectangle(
        frame_copy,
        (ROI_left,ROI_top),
        (ROI_right,ROI_bottom),
        (255,128,0),
        2
    )

    cv2.imshow(
        "Sign Language Typing",
        frame_copy
    )

    # ======================
    # KEYBOARD
    # ======================

    key=cv2.waitKeyEx(20)

    # SPACE -> add character
    if key==32:

        if current_prediction=="CLEAR":
            sentence=""

        elif current_prediction!="":
            sentence+=current_prediction


    # BACKSPACE -> delete
    elif key==8 or key==2555904:
        sentence=sentence[:-1]


    # ESC -> exit
    elif key==27:
        break


cam.release()
cv2.destroyAllWindows()