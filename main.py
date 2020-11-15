import cv2 as cv

from ml import Counter

TEST_IMAGE = True
TEST_VIDEO = False

if __name__ == "__main__":
    # Download "yolov3-spp.weights" at https://pjreddie.com/media/files/yolov3-spp.weights
    counter = Counter("yolov3-spp.cfg", "yolov3-spp.weights", "coco.names")

    if TEST_IMAGE:
        frame = cv.imread('test.jpg')
        boxes, indexes = counter.apply_yolo(frame)
        frame = counter.draw_object(frame, boxes, indexes)
        cv.imwrite('test_result.jpg', frame)

    if TEST_VIDEO:
        cap = cv.VideoCapture('test_vid.avi')
        w, h = int(cap.get(3)), int(cap.get(4))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            boxes, indexes = counter.apply_yolo(frame)
            frame = counter.draw_object(frame, boxes, indexes)

            cv.imshow("test_video", frame)
            cv.waitKey(0)

        cap.release()
        cv.destroyAllWindows()
