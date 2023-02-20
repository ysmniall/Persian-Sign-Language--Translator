import cv2
import mediapipe as mp
import math

def hand_crop(img, image_size=(480, 480), zoomout_ratio=2e-2):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=10)
    
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(imgRGB)

    crop_BGR = []

    if results.multi_hand_landmarks:
        for hlm in results.multi_hand_landmarks:
            x_max = int(max([i.x for i in hlm.landmark]) * image_size[0])
            y_max = int(max([i.y for i in hlm.landmark]) * image_size[1])
            x_min = int(min([i.x for i in hlm.landmark]) * image_size[0])
            y_min = int(min([i.y for i in hlm.landmark]) * image_size[1])

            width, height = x_max - x_min, y_max - y_min

            if width > height:
                y_min -= (((width - height) / 2) + image_size[1] * zoomout_ratio)
                y_min = math.floor(y_min)
                y_max += (((width - height) / 2) + image_size[1] * zoomout_ratio)
                y_max = math.floor(y_max)
                x_min -= (image_size[0] * zoomout_ratio)
                x_min = math.floor(x_min)
                x_max += (image_size[0] * zoomout_ratio)
                x_max = math.floor(x_max)
            else:
                x_min -= (((height - width) / 2) + image_size[0] * zoomout_ratio)
                x_min = math.floor(x_min)
                x_max += (((height - width) / 2) + image_size[0] * zoomout_ratio)
                x_max = math.floor(x_max)
                y_min -= (image_size[1] * zoomout_ratio)
                y_min = math.floor(y_min)
                y_max += (image_size[1] * zoomout_ratio)
                y_max = math.floor(y_max)

            crop = img[y_min:y_max, x_min:x_max, :]
            cropBGR = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            crop_BGR.append(cropBGR)
            
    return crop_BGR