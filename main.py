import os

import numpy as np
from PIL import Image
from tensorflow.compat.v1 import logging

from ml import CameraSegmentator
from ml.Mask_RCNN.coco import CocoConfig

COCO_WEIGHTS_PATH = "weights/mask_rcnn_coco.h5"
GPU_COUNT = 1
IMAGES_PER_GPU = 1

IMAGES_DIR = "test_vid_frames"

FRAME_DELTA = 10
VIDEO_PATH = "test_vid.avi"
VIDEO_PREDICT_DIR = "video/"

if not os.path.exists(VIDEO_PREDICT_DIR):
    os.mkdir(VIDEO_PREDICT_DIR)

logging.set_verbosity(logging.ERROR)


# Create configuration
class InferenceConfig(CocoConfig):
    GPU_COUNT = GPU_COUNT
    IMAGES_PER_GPU = IMAGES_PER_GPU


if __name__ == '__main__':
    ml = CameraSegmentator(COCO_WEIGHTS_PATH)  # , InferenceConfig)  # FIXME

    images_paths = sorted([path for path in os.listdir(IMAGES_DIR) if path.endswith(".png")])
    images = [np.array(Image.open(os.path.join(IMAGES_DIR, path)).convert("RGB")) for path in images_paths]

    preds = ml.predict(images)
    visualized = ml.visualize(images, preds)
    for image, path in zip(visualized, images_paths):
        image_PIL = Image.fromarray(image)
        image_PIL.save(os.path.join(VIDEO_PREDICT_DIR, path))

    print(visualized)
