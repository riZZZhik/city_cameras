import os
import cv2 as cv

from counter import Counter

VIDEO_FROM_FRAMES = False
VIDEO_PATH = "test_vid.avi"

if __name__ == "__main__":
    points = ((10, 600), (1900, 460))
    counter = Counter(points, "yolo_files/yolov3-spp.cfg", "yolo_files/yolov3-spp.weights", "yolo_files/coco.names",
                      classes=("person", "car", "bus", "bicycle", "motorbike", "truck"), show_processed_frame=True)

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
        output = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20, size)

        i = 0
        while cap.isOpened():
            i += 1
            ret, frame = cap.read()
            if not ret:
                break

            counted, processed_frame = counter.count(frame)
            text_count = [f"{x}: {i}" for x, i in counted.items()]
            print(f"Counted: {', '.join(text_count)}, on frame {i}")
            output.write(processed_frame)

        cap.release()
        output.release()

    counter.save_to_json("output.json", "test_vid")
