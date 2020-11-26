import os
import cv2 as cv

from counter import Counter
from config import VIDEO_PATH, OUTPUT_VIDEO_PATH, OUTPUT_JSON_PATH, POINTS, CLASSES, YOLO_FILES_PATH
assert OUTPUT_VIDEO_PATH.endswith("mp4"), "Output video supports only MP4"

VIDEO_FROM_FRAMES = False

if __name__ == "__main__":

    counter = Counter(POINTS, YOLO_FILES_PATH, classes=CLASSES, show_processed_frame=OUTPUT_VIDEO_PATH)

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
            output = cv.VideoWriter(OUTPUT_VIDEO_PATH, cv.VideoWriter_fourcc(*'MP4V'), 20, size)

        i = 0
        while cap.isOpened():
            i += 1
            ret, frame = cap.read()
            if not ret:
                break

            counted, processed_frame = counter.count(frame)
            text_count = [f"{x}: {i}" for x, i in counted.items()]
            print(f"Counted: {', '.join(text_count)}, on frame {i}")
            if OUTPUT_VIDEO_PATH:
                output.write(processed_frame)

        cap.release()
        output.release()

    if OUTPUT_JSON_PATH:
        counter.save_to_json(OUTPUT_JSON_PATH, "test_vid")
