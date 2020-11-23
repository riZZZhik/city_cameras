import cv2 as cv
import json
import numpy as np
import os


class Counter:  # TODO: Return number of each object type
    """Counter object to segment, outline and count objects on camera

    :param points: Two points to draw line, like ((x1, y1), (x2, y2))
    :type points: list or tuple
    :param yolo_cfg: Path to "yolo3-spp.cfg"
    :type yolo_cfg: str
    :param yolo_weights: Path to "yolo3-spp.weights" from https://pjreddie.com/media/files/yolov3-spp.weights
    :type yolo_weights: str
    :param coco_names: Path to "coco_names"
    :type coco_names: str
    :param classes: Coco classes to count
    :type classes: list or tuple
    :param processed_frame: Should count func return processed image?
    :type processed_frame: bool
    """

    def __init__(self, points, yolo_cfg, yolo_weights, coco_names, classes=None, processed_frame=False):
        assert np.array(points).shape == (2, 2), "Points var should be like ((x1, y1), (x2, y2))"

        # Import coco classes
        with open(coco_names) as f:
            self.coco_classes = {i: x for i, x in enumerate(f.read().split('\n'))}

        # Class variables
        self.centroids = []
        if classes is None:
            classes = ("person", "car", "bus", "bicycle", "motorbike")
        else:
            assert type(classes) in (list, tuple)
        self.classes = {i: x for i, x in self.coco_classes.items() if x in classes}
        print(f'Used "{", ".join(self.classes.values())}" classes')
        self.counted = {x: 0 for x in classes}

        self.points = points
        self.processed_frame = processed_frame

        # Init YOLO
        self.net = cv.dnn.readNet(yolo_cfg, yolo_weights)
        layer_names = self.net.getLayerNames()
        out_layers_indexes = self.net.getUnconnectedOutLayers()
        self.out_layers = [layer_names[index[0] - 1] for index in out_layers_indexes]

    def count(self, frame: np.array):
        """Function to apply YOLOv3 NN to segment COCO objects

        :param frame: Image array
        :type frame: np.ndarray

        :return len(self.centroids): Number of counted objects
        :rtype len(self.centroids): int
        :return processed_frame: Processed frame with outlined objects, if init processed_frame is True
        :rtype processed_frame: np.ndarray
        """
        height, width = frame.shape[:2]
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_layers)

        boxes = []
        class_indexes = []
        class_scores = []
        for out in outs:
            for obj in out:
                scores = obj[5:]
                class_index = np.argmax(scores)
                class_score = scores[class_index]
                if class_score > 0 and class_index in self.classes.keys():
                    cx = int(obj[0] * width)
                    cy = int(obj[1] * height)
                    obj_width = int(obj[2] * width)
                    obj_height = int(obj[3] * height)
                    x = cx - obj_width // 2
                    y = cy - obj_height // 2
                    box = [x, y, obj_width, obj_height]
                    boxes.append(box)
                    class_indexes.append(class_index)
                    class_scores.append(float(class_score))

        chosen_boxes = cv.dnn.NMSBoxes(boxes, class_scores, 0.2, 0.5)

        if self.processed_frame:
            processed_frame = frame.copy()

        sector = lambda x, y: (x - self.points[0][0]) * (self.points[1][1] - self.points[0][1]) - \
                              (y - self.points[0][1]) * (self.points[1][0] - self.points[0][0])

        for box_index in chosen_boxes:
            index = box_index[0]
            obj_class = self.classes[class_indexes[index]]
            x, y, w, h = boxes[index]
            cx = x + w // 2
            cy = y + h // 2
            end = (x + w, y + h)
            color = (255, 255, 255)
            start = (x - 10, y - 10)

            if self.processed_frame:
                processed_frame = cv.rectangle(processed_frame, start, end, color, 2)
                processed_frame = cv.putText(processed_frame, obj_class, start, cv.FONT_HERSHEY_SIMPLEX,
                                             1, color, 2, cv.LINE_AA)

            if abs(sector(cx, cy)) < 10000:
                if len(self.centroids) == 0:
                    self.centroids.append((cx, cy))
                    self.counted[obj_class] += 1
                for i in range(len(self.centroids)):
                    dist = cv.norm(self.centroids[i], (cx, cy), cv.NORM_L2)
                    if dist < 20:  # TODO: Distance coefficient
                        self.centroids[i] = (cx, cy)
                        break
                    i += 1
                    if i == len(self.centroids):
                        self.centroids.append((cx, cy))
                        self.counted[obj_class] += 1

        if self.processed_frame:
            text_count = [f"{x}: {i}" for x, i in self.counted.items()]
            text_count = 'Counted: ' + ', '.join(text_count)
            processed_frame = cv.line(processed_frame, *self.points, (0, 255, 0), 2)
            processed_frame = cv.putText(processed_frame, text_count, (10, height - 10), cv.FONT_HERSHEY_SIMPLEX,
                                         1, (255, 255, 255), 2, cv.LINE_AA)
            return self.counted, processed_frame
        else:
            return self.counted, frame

    def save_to_json(self, filename, camera_id):
        data = {}
        if os.path.exists(filename):
            with open(filename) as f:
                previous_data = json.load(f)
                for key, value in previous_data.items():
                    data[key] = value

        data[camera_id] = self.counted

        with open(filename, 'w') as f:
            json.dums(data, f)
