import torch
import time
import numpy as np
from numpy import expand_dims
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model


device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((128, 128)),
                            transforms.ToTensor()
                           ])

nomask_model = InceptionResnetV1(pretrained='vggface2').eval() # Model of facenet for normal face recognition

top_model = load_model(r'models\top_patch.h5')
top_model = Model(inputs= top_model.inputs, outputs=top_model.layers[-2].output)
mask_model = top_model

# tr_model = load_model(r'models\top_right_patch.h5')
# tr_model = Model(inputs= tr_model.inputs, outputs=tr_model.layers[-2].output)

# tl_model = load_model(r'models\top_left_patch.h5')
# tl_model = Model(inputs= tl_model.inputs, outputs=tl_model.layers[-2].output)


def get_embedding(model, face_pixels):
    samples = expand_dims(face_pixels, axis=0)    
    yhat = model.predict(samples)
    return yhat[0]

def recognize_mask(face, threshold):
    # start = time.time()
    data = np.load(r'database\mask_embeddings.npz')
    local_embeds, names = data['arr_0'], data['arr_1']    
    embedding = get_embedding(mask_model, face)

    distances = list()
    for ebd in local_embeds:
        distance = np.sum(np.square(embedding - ebd))
        distances.append(distance)
    # print(distances)    
    min_dist = min(distances)
    min_idx = distances.index(min_dist)
    min_dist = min_dist/100
    # print(min_dist)    

    if min_dist > threshold:
        name = 'Unknown'
    else:
        name = names[min_idx]
    return min_dist, name   

def recognize_nomask(face, threshold):
    # start = time.time()
    local_embeds = torch.load(r'database\nomask_ebd.pth')
    names = np.load(r'database\nomask_usernames.npy')
    
    nomask = trans(face).float().unsqueeze(0)
    # nomask = norm(nomask)
    with torch.no_grad():
        embed = nomask_model(nomask)
    diff = embed.flatten() - local_embeds.flatten(start_dim=1)
    norml2 = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1))
    min_dist, idx = torch.min(norml2, dim=0)

    if min_dist > threshold:
        name = 'Unknown'        
    else:
        name = names[idx]
    # print('time :', time.time()- start)
    return min_dist, name  