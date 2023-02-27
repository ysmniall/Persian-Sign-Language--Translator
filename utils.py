import cv2
import mediapipe as mp
import math

def hand_crop(img, zoomout_ratio=2e-2):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=10)
    
    x = img.shape[1]
    y = img.shape[0]

    zoomout_x = x * zoomout_ratio
    zoomout_y = y * zoomout_ratio

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_padding = cv2.copyMakeBorder(imgRGB, math.ceil(zoomout_y), math.ceil(zoomout_y), math.ceil(zoomout_x), math.ceil(zoomout_x), cv2.BORDER_CONSTANT, (0, 0, 0))

    results = hands.process(img_padding)

    crop_result = []

    if results.multi_hand_landmarks:
        for hlm in results.multi_hand_landmarks:
            x_max = int(max([i.x for i in hlm.landmark]) * (x + 2 * zoomout_x))   #0 ta 1 mide * 480
            y_max = int(max([i.y for i in hlm.landmark]) * (y + 2 * zoomout_y))
            x_min = int(min([i.x for i in hlm.landmark]) * (x + 2 * zoomout_x))
            y_min = int(min([i.y for i in hlm.landmark]) * (y + 2 * zoomout_y))

            width, height = x_max - x_min, y_max - y_min

            if width > height:
                y_min -= (((width - height) / 2) + zoomout_y)
                y_min = math.floor(y_min)
                y_max += (((width - height) / 2) + zoomout_y)
                y_max = math.floor(y_max)
                x_min -= zoomout_x
                x_min = math.floor(x_min)
                x_max += zoomout_x
                x_max = math.floor(x_max)
            else:
                x_min -= (((height - width) / 2) + zoomout_x)
                x_min = math.floor(x_min)
                x_max += (((height - width) / 2) + zoomout_x)
                x_max = math.floor(x_max)
                y_min -= zoomout_y
                y_min = math.floor(y_min)
                y_max += zoomout_y
                y_max = math.floor(y_max)

            crop = img_padding[y_min:y_max, x_min:x_max, :]
            cropBGR = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            crop_result.append(cropBGR)
            
    return crop_result