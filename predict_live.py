# import cv2
# import numpy as np
# from tensorflow import keras
# from googletrans import Translator
# from PIL import Image, ImageDraw, ImageFont
# import os

# # ======================
# # LOAD MODEL
# # ======================

# if not os.path.exists("best_model.h5"):
#     print("Model not found")
#     exit()

# model = keras.models.load_model("best_model.h5")

# translator = Translator()

# print("Model Loaded Successfully")


# # ======================
# # LABELS
# # ======================

# labels = {
# 0:'A',
# 1:'B',
# 2:'C',
# 3:'CLEAR',
# 4:'D',
# 5:'DELETE',
# 6:'E',
# 7:'F',
# 8:'G',
# 9:'H',
# 10:'I',
# 11:'J',
# 12:'K',
# 13:'L',
# 14:'M',
# 15:'N',
# 16:'O',
# 17:'P',
# 18:'Q',
# 19:'R',
# 20:'S',
# 21:'SPACE',
# 22:'T',
# 23:'U',
# 24:'V',
# 25:'W',
# 26:'X',
# 27:'Y',
# 28:'Z'
# }

# ROI_top = 100
# ROI_bottom = 300
# ROI_right = 150
# ROI_left = 350

# sentence = ""
# hindi_text = ""
# current_prediction = ""


# # ======================
# # HAND SEGMENTATION
# # ======================

# def segment_hand(frame):

#     _, thresh = cv2.threshold(
#         frame,
#         0,
#         255,
#         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#     )

#     kernel = np.ones((5,5),np.uint8)

#     thresh = cv2.morphologyEx(
#         thresh,
#         cv2.MORPH_OPEN,
#         kernel
#     )

#     thresh = cv2.morphologyEx(
#         thresh,
#         cv2.MORPH_CLOSE,
#         kernel
#     )

#     contours,_ = cv2.findContours(
#         thresh.copy(),
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE
#     )

#     if len(contours)==0:
#         return None

#     hand=max(
#         contours,
#         key=cv2.contourArea
#     )

#     if cv2.contourArea(hand)<5000:
#         return None

#     return thresh,hand


# # ======================
# # CAMERA
# # ======================

# cam=cv2.VideoCapture(0)

# while True:

#     ret,frame=cam.read()

#     if not ret:
#         break

#     frame=cv2.flip(frame,1)

#     frame_copy=frame.copy()

#     roi=frame[
#         ROI_top:ROI_bottom,
#         ROI_right:ROI_left
#     ]

#     gray=cv2.cvtColor(
#         roi,
#         cv2.COLOR_BGR2GRAY
#     )

#     gray=cv2.GaussianBlur(
#         gray,
#         (7,7),
#         0
#     )

#     hand=segment_hand(gray)

#     if hand is not None:

#         thresholded,hand_segment=hand

#         cv2.drawContours(
#             frame_copy,
#             [hand_segment+(ROI_right,ROI_top)],
#             -1,
#             (255,0,0),
#             2
#         )

#         cv2.imshow(
#             "Threshold",
#             thresholded
#         )

#         img=cv2.resize(
#             thresholded,
#             (64,64)
#         )

#         img=cv2.cvtColor(
#             img,
#             cv2.COLOR_GRAY2RGB
#         )

#         img=img/255.0

#         img=np.reshape(
#             img,
#             (1,64,64,3)
#         )

#         prediction=model.predict(
#             img,
#             verbose=0
#         )

#         index=np.argmax(prediction)

#         confidence=np.max(prediction)

#         if confidence>0.90:

#             current_prediction=labels[index]


#     # ======================
#     # UI
#     # ======================

#     cv2.putText(
#         frame_copy,
#         "Current: "+current_prediction,
#         (20,50),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0,0,255),
#         2
#     )

#     cv2.putText(
#         frame_copy,
#         "TEXT: "+sentence,
#         (20,400),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0,255,0),
#         2
#     )

#     cv2.rectangle(
#         frame_copy,
#         (ROI_left,ROI_top),
#         (ROI_right,ROI_bottom),
#         (255,128,0),
#         2
#     )

#     # ======================
#     # Hindi Display
#     # ======================

#     pil_img=Image.fromarray(frame_copy)

#     draw=ImageDraw.Draw(pil_img)

#     font=ImageFont.truetype(
#         "C:/Windows/Fonts/mangal.ttf",
#         28
#     )

#     draw.rectangle(
#         [(10,420),(900,510)],  
#         fill=(0,0,0)
#     )

#     draw.text(
#         (20,445),
#         "Hindi: "+hindi_text,
#         font=font,
#         fill=(0,255,255)
#     )

#     frame_copy=np.array(pil_img)

#     cv2.imshow(
#         "Sign Language Typing",
#         frame_copy
#     )

#     # ======================
#     # KEYS
#     # ======================

#     key=cv2.waitKeyEx(20)

#     # SPACE → Add character

#     if key==32:

#         if current_prediction=="CLEAR":

#             sentence=""
#             hindi_text=""

#         elif current_prediction!="":

#             sentence+=current_prediction


#     # BACKSPACE → delete

#     elif key==8:

#         sentence=sentence[:-1]


#     # ENTER → Hindi translate

#     elif key==13:

#         try:

#             if sentence!="":

#                 hindi_text=translator.translate(
#                     sentence,
#                     dest='hi'
#                 ).text

#         except:

#             hindi_text="Translation Error"


#     # ESC

#     elif key==27:
#         break


# cam.release()
# cv2.destroyAllWindows()


# ======================= New UI =================== #
import cv2
import numpy as np
from tensorflow import keras
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont
import os
import time

# =========================
# LOAD MODEL
# =========================

if not os.path.exists("best_model.h5"):
    print("Model not found")
    exit()

model = keras.models.load_model("best_model.h5")
translator = Translator()

print("Model Loaded Successfully")


# =========================
# LABELS
# =========================

labels = {
0:'A',1:'B',2:'C',3:'CLEAR',4:'D',
5:'DELETE',6:'E',7:'F',8:'G',9:'H',
10:'I',11:'J',12:'K',13:'L',14:'M',
15:'N',16:'O',17:'P',18:'Q',19:'R',
20:'S',21:'SPACE',22:'T',23:'U',
24:'V',25:'W',26:'X',27:'Y',28:'Z'
}


# =========================
# SETTINGS
# =========================

ROI_top=120
ROI_bottom=340
ROI_right=150
ROI_left=370

sentence=""
hindi_text=""

current_prediction="0"
confidence=0.0

prev_time=time.time()


# =========================
# HAND SEGMENT
# =========================

def segment_hand(frame):

    _,thresh=cv2.threshold(
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


# =========================
# CAMERA
# =========================

cam=cv2.VideoCapture(0)

cam.set(3,1280)
cam.set(4,720)


while True:

    ret,frame=cam.read()

    if not ret:
        break

    frame=cv2.flip(frame,1)

    frame_copy=frame.copy()


    # FPS

    current=time.time()

    fps=int(
        1/(current-prev_time)
    )

    prev_time=current


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

    current_prediction="0"
    confidence=0.0


    # =========================
    # HAND FOUND
    # =========================

    if hand is not None:

        thresholded,hand_segment=hand

        cv2.drawContours(
            frame_copy,
            [hand_segment+(ROI_right,ROI_top)],
            -1,
            (255,128,0),
            3
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

        temp_confidence=float(
            np.max(prediction)
        )*100


        if temp_confidence>90:

            current_prediction=labels[index]
            confidence=temp_confidence

        else:

            current_prediction="0"
            confidence=0.0


    # =========================
    # TOP PANEL
    # =========================

    cv2.rectangle(
        frame_copy,
        (0,0),
        (1280,100),
        (30,30,30),
        -1
    )


    cv2.putText(
        frame_copy,
        f"Current: {current_prediction}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )


    cv2.putText(
        frame_copy,
        f"Confidence: {confidence:.1f}%",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )


    cv2.putText(
        frame_copy,
        f"FPS: {fps}",
        (450,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )


    # ROI box

    cv2.rectangle(
        frame_copy,
        (ROI_right,ROI_top),
        (ROI_left,ROI_bottom),
        (255,128,0),
        3
    )


    # =========================
    # BOTTOM PANEL
    # =========================

    cv2.rectangle(
        frame_copy,
        (0,500),
        (1280,720),
        (30,30,30),
        -1
    )


    cv2.putText(
        frame_copy,
        "TEXT:",
        (20,550),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )


    cv2.putText(
        frame_copy,
        sentence,
        (180,550),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )


    # Hindi text

    pil=Image.fromarray(frame_copy)

    draw=ImageDraw.Draw(pil)

    font=ImageFont.truetype(
        "C:/Windows/Fonts/mangal.ttf",
        28
    )


    draw.text(
        (20,600),
        "Hindi:",
        font=font,
        fill=(0,255,255)
    )

    draw.text(
        (180,600),
        hindi_text,
        font=font,
        fill=(255,255,255)
    )

    frame_copy=np.array(pil)

    cv2.imshow(
        "Sign Language Typing",
        frame_copy
    )


    # =========================
    # KEYBOARD
    # =========================

    key=cv2.waitKeyEx(20)


    # Space key pressed
    if key==32:

        if current_prediction=="CLEAR":

            sentence=""
            hindi_text=""

        elif current_prediction=="DELETE":

            sentence=sentence[:-1]

        elif current_prediction=="SPACE":

            sentence+=" "

        elif current_prediction!="0":

            sentence+=current_prediction


    # keyboard backspace
    elif key==8:

        sentence=sentence[:-1]


    # ENTER translate
    elif key==13:

        try:

            if sentence!="":

                hindi_text=translator.translate(
                    sentence,
                    dest='hi'
                ).text

        except:

            hindi_text="Translation Error"


    # ESC exit
    elif key==27:

        break

cam.release()
cv2.destroyAllWindows()