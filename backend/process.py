import numpy as np
import PIL.Image as Image
import io
import cv2

def preprocess(array):
    if len(array) != 0:
        image_array= Image.open(io.BytesIO(array)).convert('RGB')
        cv2_image=cv2.cvtColor(np.array(image_array), cv2.COLOR_BGR2RGB)
        cv2_image = cv2.resize(cv2_image, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
        return cv2_image
    return None
        