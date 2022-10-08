import cv2 
from PIL import Image
import os
import numpy as np

# function to patch the face image 
def patch_faces(image, filename, i):
    image = cv2.imread(image)
    # folders of the patched dataset
    folder_save_top = r'FEI\top\test'
    folder_save_topl = r'FEI\top_left\test'
    folder_save_topr = r'FEI\top_right\test'
    if image is not None:

        height, width, channels = image.shape
        crop_bot = image[int(height / 2):height, 0:width]
        crop_top = image[0:int(height / 2) + 10, 0:width]

        crop_top = Image.fromarray(crop_top)
        crop_top = crop_top.resize((160,160))
        save_patches(crop_top, filename, i, folder_save_top)

        top_left = image[0:int(height / 2) + 10, 0:int(width / 2)]
        top_left = Image.fromarray(top_left)
        top_left = top_left.resize((160, 160))
        save_patches(top_left, filename, i, folder_save_topl)

        top_right = image[0:int(height / 2) + 10, int(width / 2):width]
        top_right = Image.fromarray(top_right)
        top_right = top_right.resize((160, 160))
        save_patches(top_right, filename, i, folder_save_topr)

# function to save the patches in different folders
def save_patches(image, filename, i, folder_save):
    folder_save = os.path.join(folder_save, i)
    save = os.path.join(folder_save, filename)
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    image = np.array(image)
    cv2.imwrite(save, image)

if __name__ == '__main__':
    folder = r'FEI\test'
    for i in os.listdir(folder):
        fold = os.path.join(folder, i)
        for filename in os.listdir(fold):
            image = os.path.join(fold, filename)
            
            patch_faces(image, filename, i)