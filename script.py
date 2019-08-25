import os
import numpy as np 
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
import requests

class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_config"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80
    DETECTION_MIN_CONFIDENCE = 0.6

def get_car_boxes(boxes, class_ids):
    car_boxes = []
    for i, box in enumerate(boxes):
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)
    return np.array(car_boxes)

BASE_DIR = Path(".")
MODEL_DIR = os.path.join(BASE_DIR,'logs')
COCO_PATH = os.path.join(BASE_DIR,"mask_rcnn_coco.h5")
VIDEO_PATH = os.path.join(BASE_DIR,"video/footage.mp4")

URL = "https://localhost:8000/api/"




if not os.path.exists(COCO_PATH):
    mrcnn.utils.download_trained_weights(COCO_PATH)

model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
model.load_weights(COCO_PATH,by_name=True)

#loading video
video = cv2.VideoCapture(VIDEO_PATH)
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))

parked_cars = None

free_space_frames = 0

while video.isOpened():
    loaded, frame = video.read()
    if not loaded:
        break
    
    rgb_image = frame[:,:,::-1]
    results = model.detect([rgb_image], verbose=0)
    res = results[0]
    if parked_cars is None:
        parked_cars = get_car_boxes(res['rois'], res['class_ids'])
    else:
        car_boxes = get_car_boxes(res['rois'], res['class_ids'])
        overlaps = mrcnn.utils.compute_overlaps(parked_cars, car_boxes)
        free_space = False
        for parking_area, overlap_areas in zip(parked_cars, overlaps):
            max_IoU_overlap = np.max(overlap_areas)
            y1, x1, y2, x2 = parking_area
            if max_IoU_overlap < 0.15:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                free_space = True
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))

        if free_space:
            free_space_frames +=1
        else:
            free_space_frames = 0
        
        if free_space_frames>10:
            print("Free Space")


    # out.write(frame)
    cv2.imshow('video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()