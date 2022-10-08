from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import os
import numpy as np
import cv2

def extract_face(filename, folder_save, detector):
    pixels = pyplot.imread(filename)
    img = cv2.imread(filename)
    results = detector.detect_faces(pixels)
    print(results)
    if not results:
        print(filename)
        return None
    else:
        # extract bbox from the first face
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        # somttimes these are negative so make them become zero
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        left_eye_center = results[0]['keypoints']['left_eye']
        right_eye_center = results[0]['keypoints']['right_eye']

        # align face before cropping

        # rotated = align_face(img, left_eye_center, right_eye_center)
        # rotated = rotated[y1:y2, x1:x2]
        # rotated = cv2.resize(rotated, (224, 224)) # theo paper (160, 160)

        # align face after cropping
        face = img[y1:y2, x1:x2]
        rotated = align_face(face, np.asarray(left_eye_center) - np.asarray((x1,y1)), np.asarray(right_eye_center) - np.asarray((x1,y1)))
        face = cv2.resize(rotated, (224, 224))
        # cv2.imshow("image", rotated)
        # cv2.waitKey(0)
        # cv2.imwrite(folder_save, face)
        return face

def align_face(img, left_eye_center, right_eye_center):
    delta_x = right_eye_center[0] - left_eye_center[0]
    delta_y = right_eye_center[1] - left_eye_center[1]

    angle = np.arctan(delta_y/ delta_x)
    angle = (angle * 180)/ np.pi

    h, w  = img.shape[0], img.shape[1]
    center = (w//2, h//2)
    
    # M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    rotated = cv2.warpAffine(img, M, (w,h))
    # cv2.imshow('image', rotated)
    # cv2.waitKey(0)
    return rotated

if __name__ == '__main__':
    folder = 'FEI'
    folder_save = 'FEI'

    detector = MTCNN()
    for i in os.listdir(folder):
        fold = os.path.join(folder, i)
        folder_s = os.path.join(folder_save, i)
        for j in os.listdir(fold):
            user = os.path.join(fold, j)
            user_s = os.path.join(folder_s, j)
            for filename in os.listdir(user):
                image = os.path.join(user, filename)
                save = os.path.join(user_s, filename)
                if image is not None:
                    pixels = extract_face(image, save, detector=detector)