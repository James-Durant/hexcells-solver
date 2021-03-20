import os, re, cv2
import numpy as np

def load_digits(digit_type):
    path = '../resources/training/'+digit_type
    images, labels = [], []
    max_x, max_y = -float('inf'), -float('inf')
    for filename in os.listdir(path):
        if filename.endswith(".png"): 
            digit = re.search(digit_type+'_digit_(.+?)_', filename).group(1)
            if digit == 'question-mark':
                digit = '?'
                
            image = cv2.imread(path+'/'+filename, cv2.IMREAD_GRAYSCALE)
            image = np.where(image == 255, 0, 1)
            max_x, max_y = max(max_x, image.shape[0]), max(max_y, image.shape[1])
            images.append(image)
            labels.append(digit)

    images = [pad_to_dims(image, max_x, max_y) for image in images]
    
    for image in images:
        cv2.imshow('test', image)
        cv2.waitKey(0)
    
    return images, labels, max_x, max_y

def pad_to_dims(arr, xx, yy):
    h = arr.shape[0]
    w = arr.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(arr, pad_width=((a, aa), (b, bb)), mode='constant', constant_values=255)

load_digits('black')