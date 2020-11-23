import cv2 as cv

from counter import Counter

if __name__ == "__main__":
    # Download "yolov3-spp.weights" at https://pjreddie.com/media/files/yolov3-spp.weights
    points = ((10, 600), (1900, 460))
    counter = Counter(points, "yolo_files/yolov3-spp.cfg", "yolo_files/yolov3-spp.weights", "yolo_files/coco.names",
                      classes=("person", "car", "bus", "bicycle", "motorbike"), processed_frame=True)

    cap = cv.VideoCapture('test_vid.avi')
    size = int(cap.get(3)), int(cap.get(4))
    output = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc(*'MJPG'), 20, size)

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

    counter.save_to_json("output.json", "test_vid")
    cap.release()
    output.release()
