import os

import cv2 as cv
from tensorflow.compat.v1 import logging

from ml import CameraSegmentator

IMAGES_DIR = "images"

FRAME_DELTA = 10
VIDEO_PATH = "test_vid.avi"
VIDEO_PREDICT_DIR = "video/"

if not os.path.exists(VIDEO_PREDICT_DIR):
    os.mkdir(VIDEO_PREDICT_DIR)

logging.set_verbosity(logging.ERROR)


if __name__ == '__main__':
    ml = CameraSegmentator("weights/mask_rcnn_coco.h5")
    cap = cv.VideoCapture(VIDEO_PATH)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            cv.imshow('Frame', frame)
            cv.waitKey(1)

            if frame_id % FRAME_DELTA == 0:
                prediction = ml.predict([frame])
                visualized = ml.visualize([frame], prediction)
                cv.imwrite(os.path.join(VIDEO_PREDICT_DIR, str(frame_id) + ".jpg"), visualized[0])

            frame_id += 1
        else:
            break

    cap.release()
    cv.destroyAllWindows()

# if __name__ == '__main__':
#     ml = CameraSegmentator("weights/mask_rcnn_coco.h5")
#
#     images = []
#     for file in os.listdir(IMAGES_DIR):
#         if file.endswith(".png") or file.endswith(".jpg"):
#             image = Image.open(os.path.join(IMAGES_DIR, file)).convert("RGB")
#             images.append(np.array(image))
#             break
#
#     r = ml.predict(images)[0]
#     result_instance = display_instances(images[0], r['rois'], r['masks'], r['class_ids'],
#                                               ml.class_names, r['scores'])
