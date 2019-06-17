import argparse
import time

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from keras.applications import ResNet50, imagenet_utils
from keras.preprocessing.image import img_to_array

from utils.simple_object_detector import image_pyramid, sliding_window, classify_batch

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', required=True, help='Path to the input image.')
argument_parser.add_argument('-c', '--confidence', type=float, default=0.5,
                             help='Minimum probability to filter weak detections.')
arguments = vars(argument_parser.parse_args())

INPUT_SIZE = (350, 350)
PYRAMID_SCALE = 2
WINDOW_STEP = 32
ROI_SIZE = (224, 224)
BATCH_SIZE = 64

print('[INFO] Loading network...')
model = ResNet50(weights='imagenet', include_top=True)

labels = dict()

original = cv2.imread(arguments['image'])
(h, w) = original.shape[:2]

resized = cv2.resize(original, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)

batch_rois = None
batch_locations = list()

print('[INFO] Detecting objects...')
start = time.time()

for image in image_pyramid(resized, scale=PYRAMID_SCALE, min_size=ROI_SIZE):
    for (x, y, roi) in sliding_window(image, WINDOW_STEP, ROI_SIZE):
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        roi = imagenet_utils.preprocess_input(roi)

        if batch_rois is None:
            batch_rois = roi
        else:
            batch_rois = np.vstack([batch_rois, roi])

        batch_locations.append((x, y))

        if len(batch_rois) == BATCH_SIZE:
            labels = classify_batch(model, batch_rois, batch_locations, labels, min_prob=arguments['confidence'])

            batch_rois = None
            batch_locations = list()

if batch_rois is not None:
    labels = classify_batch(model, batch_rois, batch_locations, labels, min_prob=arguments['confidence'])

end = time.time()

print(f'[INFO] Detections took {end - start:.4f} seconds.')

for k in labels.keys():
    clone = resized.copy()

    for (box, prob) in labels[k]:
        (x_start, y_start, x_end, y_end) = box
        cv2.rectangle(clone, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    cv2.imshow('Without NMS', clone)
    clone = resized.copy()

    boxes = np.array([p[0] for p in labels[k]])
    proba = np.array([p[1] for p in labels[k]])
    boxes = non_max_suppression(boxes, proba)

    for (x_start, y_start, x_end, y_end) in boxes:
        cv2.rectangle(clone, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

    print(f'[INFO] {k}: {len(boxes)}')
    cv2.imshow('With NMS', clone)
    cv2.waitKey(0)
