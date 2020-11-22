import cv2 as cv

from ml import Counter

if __name__ == "__main__":
    # Download "yolov3-spp.weights" at https://pjreddie.com/media/files/yolov3-spp.weights
    points = ((10, 600), (1900, 460))
    counter = Counter(points, "yolov3-spp.cfg", "yolov3-spp.weights", "coco.names", processed_frame=True)

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
        print(f"Counted: {counted}, on frame {i}")
        output.write(processed_frame)

    cap.release()
    output.release()
