import cv2
import time
from myutils.recognize_func import recognize_mask, recognize_nomask
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
from extraction2 import align_face
from detect_face import inference # function to detect mask 
from create_embeddings import patch_faces


detector = MTCNN()

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

score = 0
name = 0

while cam.isOpened():
    start = time.time()
    ret, frame = cam.read()
    if ret:        
        image = frame.copy()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(frame) # detect faces
        masks = inference(frame, conf_thresh= 0.5, iou_thresh= 0.5, input_shape= (260,260), draw_result=False) # detect mask on the faces

        if results is not None and len(results) <= len(masks) :

            for i in range(len(results)):
                # extract bbox from the first face
                x1, y1, width, height = results[i]['box']
                x2, y2 = x1 + width, y1 + height
                # somttimes these are negative so make them become zero
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                # print(x1, y1, x2, y2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (211,211,211), 2)
                if masks[i][0] ==0:
                    cv2.putText(frame, 'masked', (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255))
                
                left_eye_center = results[0]['keypoints']['left_eye']
                right_eye_center = results[0]['keypoints']['right_eye']

                # align face after cropping
                face = image[y1:y2, x1:x2]
                rotated = align_face(face, np.asarray(left_eye_center) - np.asarray((x1,y1)), np.asarray(right_eye_center) - np.asarray((x1,y1)))
                face = cv2.resize(rotated, (224, 224))
                if masks[i][0] == 1:                    
                    min_dist, name = recognize_nomask(face, threshold=0.9)
                    cv2.putText(frame, str(name), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0))
                    cv2.putText(frame, "{:.1f}".format(min_dist.item()), (x2-25, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
                if masks[i][0] == 0:
                    face = patch_faces(face)
                    min_dist, name = recognize_mask(face, threshold=1.5)
                    cv2.putText(frame, str(name), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0))
                    cv2.putText(frame, "{:.1f}".format(min_dist), (x2-25, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

        cv2.imshow('frame', frame)
        if cv2.waitKey(1)&0xFF == 27:
            break