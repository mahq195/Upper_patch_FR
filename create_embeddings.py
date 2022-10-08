from extraction2 import align_face, extract_face

import os
import torch
import cv2
import glob
import numpy as np
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from tensorflow.keras.models import load_model
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Model
# from myutils.recognize_func import recognize_nomask
from mtcnn.mtcnn import MTCNN
from PIL import Image

from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

nomask_model = InceptionResnetV1(pretrained='vggface2').eval()

top_model = load_model(r'models\top_patch.h5')
top_model = Model(inputs=top_model.inputs, outputs=top_model.layers[-2].output)
mask_model = top_model

trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((128,128)),
                            transforms.ToTensor()
                           ])

norm_layer = keras.layers.experimental.preprocessing.Normalization()


# tl_model = load_model(r'models\top_left_patch.h5')
# tl_model = Model(inputs=tl_model.inputs, outputs=tl_model.layers[-2].output)

# tr_model = load_model(r'models\top_right_patch.h5')
# tr_model = Model(inputs=tr_model.inputs, outputs=tr_model.layers[-2].output)



def get_embedding(model, face_pixels):
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]


def create_nomask_ebd(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = trans(img).float().unsqueeze(0)
    # img = norm(img).unsqueeze(0)
    with torch.no_grad():
        embed = nomask_model(img)
            

    # torch.save(torch.stack(embeddings), db_path+"/nomask_ebd.pth")
    # np.save(db_path+"/nomask_usernames", np.array(names))
    # print('Done nomask!')
    return embed

def patch_faces(image):
    # image = cv2.imread(image)/
    # folders of the patched dataset
    # folder_save_top = r'FEI\top\test'
    # folder_save_topl = r'FEI\top_left\test'
    # folder_save_topr = r'FEI\top_right\test'
    if image is not None:

        height, width, channels = image.shape
        crop_bot = image[int(height / 2):height, 0:width]
        crop_top = image[0:int(height / 2) + 10, 0:width]

        crop_top = Image.fromarray(crop_top)
        crop_top = crop_top.resize((160,160))

        return crop_top


    
if __name__ == '__main__':
    detector = MTCNN()
    data_path = r'input_image'

    mask_embeddings = list()
    mask_names = list()
    no_mask_embeddings = list()
    no_mask_names = list()

    for user in os.listdir(data_path):
        user_path = os.path.join(data_path, user)
        for mask in os.listdir(user_path): # mask or no_mask
            if mask == 'mask':
                mask_path = os.path.join(user_path, mask)  
                for image in os.listdir(mask_path):
                    file_img = os.path.join(mask_path, image)
                    print(file_img)
                    face = extract_face(file_img,folder_save=None, detector=detector)
                    if face is None:
                        continue
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = patch_faces(face)
                    # face.show()
                    embeddings = get_embedding(mask_model, face_pixels=face)
                    mask_embeddings.append(embeddings)
                    mask_names.append(user)
            else:  # no_mask
                mask_path = os.path.join(user_path, mask)  
                for image in os.listdir(mask_path):
                    file_img = os.path.join(mask_path, image)
                    print(file_img)
                    face = extract_face(file_img,folder_save=None, detector=detector)
                    if face is None:
                        continue
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    
                    # Image.fromarray(face).show()
                    embeddings = create_nomask_ebd(face)
                    no_mask_embeddings.append(embeddings)
                    no_mask_names.append(user)
    
    mask_embeddings = np.asarray(mask_embeddings)
    mask_names = np.asarray(mask_names)

    savez_compressed('database/mask_embeddings.npz', mask_embeddings, mask_names)
    # savez_compressed('database/no_mask_embeddings.npz', no_mask_embeddings, no_mask_names)
    db_path = r'database'
    torch.save(torch.stack(no_mask_embeddings), db_path+"/nomask_ebd.pth")
    np.save(db_path+"/nomask_usernames", np.array(no_mask_names))



