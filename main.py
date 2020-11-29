import os
import cv2 as cv

from counter import Counter
from config import *
assert OUTPUT_VIDEO_PATH.lower().endswith("avi"), "Output video supports only avi"

VIDEO_FROM_FRAMES = False

if __name__ == "__main__":

    counter = Counter(POINTS, yolo_dir=YOLO_FILES_DIR, log_file=LOG_FILE,
                      classes=CLASSES, show_processed_frame=bool(OUTPUT_VIDEO_PATH))

    if VIDEO_FROM_FRAMES:
        paths = [os.path.join(VIDEO_PATH, x) for x in os.listdir(VIDEO_PATH)
                 if all([x.endswith(e) for e in ['.jpg', '.png']])]

        for i, path in enumerate(paths):
            frame = cv.imread(path)

            counted, processed_frame = counter.count(frame)
            text_count = [f"{x}: {i}" for x, i in counted.items()]
            print(f"Counted: {', '.join(text_count)}, on frame {i}")

            cv.imwrite(os.path.join(VIDEO_PATH, "result", f"frame_{i}.jpg"))
    else:
        cap = cv.VideoCapture(VIDEO_PATH)
        size = int(cap.get(3)), int(cap.get(4))
        if OUTPUT_VIDEO_PATH:
            output = cv.VideoWriter(OUTPUT_VIDEO_PATH, cv.VideoWriter_fourcc(*'MJPG'), 20, size)

        i = 0
        while cap.isOpened():
            i += 1
            ret, frame = cap.read()
            if not ret:
                break

            counted, processed_frame = counter.count(frame)
            if OUTPUT_VIDEO_PATH:
                output.write(processed_frame)

        cap.release()
        output.release()

    if OUTPUT_JSON_PATH:
        counter.save_to_json(OUTPUT_JSON_PATH, CAMERA_ID)
