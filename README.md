# Upper_patch_FR
This repository is used for developing a face recognition system, even when you are wearing a face mask.
![image](https://user-images.githubusercontent.com/73461122/194694282-6261e2c3-857b-4e42-a85d-27c265b0173c.png)


## Prerequisite
1. Using `python3.7` is recommend, or you can create a virtual env with python version 3.7

2. Run this command to install packages
```
pip install -r requirements.txt
```
3. Download our pretrained model on [GDrive](https://drive.google.com/drive/folders/1--1mLFOeOoy4Mrkmw0VzVTtugizXrp0_?usp=sharing) then put them into `models` folder like `models/top_patch.h5` and `models/face_mask_detection.pb`
## How to use
1. Data preparation

For the first time, if you want to add your own images data, just prepare and add the data following the format
```
input_image/
      - giang/
            + mask/
                  ~ mask_img1
                  ~ mask_img2
                  ~ mask_img3
            + no_mask/
                  ~ no_mask_img1
                  ~ no_mask_img2
                  ~ no_mask_img3
      - linh/
            + mask/
                  ~ mask_img1
                  ~ mask_img2
                  ~ mask_img3
            + no_mask/
                  ~ no_mask_img1
                  ~ no_mask_img2
                  ~ no_mask_img3
      ...
```
Please create your own folder following the given format.

2. Create database
Run the following command to create embeddings saved inside `database` folder 
```
python create_embeddings.py
```
3. Run `camera.py` to activate recognition process using your camera
``` 
python camera.py
```

