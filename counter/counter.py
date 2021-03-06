import json
import logging
import os

import cv2 as cv
import numpy as np

from .utils import init_yolo, init_logger


class Counter:  # TODO: Return number of each object type
    """Counter object to segment, outline and count objects on camera

    :param lines: Two points to draw line, like (((x1, y1), (x2, y2)), (x3, y3), (x4, y4))
    :type lines: list or tuple
    :param yolo_dir: Path to yolo files folder
    :type yolo_dir: str
    :param yolo_paths: Paths to yolo files: "coco.names", "yolo3-spp.cfg", "yolo3-spp.weights"
    :type yolo_paths: dict
    :param dist_coef: Distance coefficient
    :type dist_coef: float
    :param classes: Coco classes to count
    :type classes: list of str or tuple of str
    :param show_processed_frame: Should count func return processed image?
    :type show_processed_frame: bool
    """

    def __init__(self, lines, yolo_dir=None, yolo_paths=None, log_file=None, log_level=logging.INFO,
                 classes=None, dist_coef=1, show_processed_frame=False):
        # TODO: User yolo_files paths
        # Init logger
        self.logger = init_logger(log_file, log_level, __name__)
        self.logger.info("Initializing Counter")

        # Assertions
        shape = np.array(lines).shape
        assert shape[-2:] == (2, 2) and len(shape) == 3, \
            "Lines variable should be like (((x1, y1), (x2, y2)), (x3, y3), (x4, y4)), ...)"

        # Init YOLOv3-SPP
        self.coco_classes, self.net, self.out_layers = init_yolo(yolo_dir, yolo_paths)

        # Classes variables
        default_classes = ("person", "car", "bus", "bicycle", "motorbike", "truck")
        if classes is None:
            classes = default_classes
        else:
            assert type(classes) in (list, tuple), "classes type should be list or tuple"
            assert all(c in default_classes for c in classes), f"Only {', '.join(default_classes)} classes supported"

        self.classes = {i: x for i, x in self.coco_classes.items() if x in classes}
        self.logger.info(f'Used "{", ".join(self.classes.values())}" classes')
        self.counted = {x: 0 for x in classes}
        self.str_counted = lambda: [f"{x}: {i}" for x, i in self.counted.items()]

        # Other class variables
        self.objects_dist = {
            "person": 1,
            "bicycle": 1.5,
            "car": 2,
            "bus": 2,
            "motorbike": 2,
            "truck": 2
        }
        self.dist_coef = dist_coef

        self.lines = lines
        self.centroids = []
        self.mins, self.maxs = [], []
        for line in lines:
            self.mins.append((min(line[0][0], line[1][0]), min(line[0][1], line[1][1])))
            self.maxs.append((max(line[0][0], line[1][0]), max(line[0][1], line[1][1])))

        self.frame_id = 0
        self.show_processed_frame = show_processed_frame

        self.logger.debug("Initialized Counter")

    def check_sector(self, x, y, sector_size):
        for line, mins, maxs in zip(self.lines, self.mins, self.maxs):
            sector = (x - line[0][0]) * (line[1][1] - line[0][1]) - (y - line[0][1]) * (line[1][0] - line[0][0])
            if abs(sector) < sector_size and \
                    (mins[0] - 5) < x < (maxs[0] + 5) and (mins[1] - 5) < y < (maxs[1] + 5):
                return True

        return False

    def count(self, frame: np.array):
        """Function to apply YOLOv3 NN to segment COCO objects

        :param frame: Image array
        :type frame: np.ndarray

        :return len(self.centroids): Number of counted objects
        :rtype len(self.centroids): int
        :return processed_frame: Processed frame with outlined objects, if init processed_frame is True
        :rtype processed_frame: np.ndarray
        """
        self.frame_id += 1
        height, width = frame.shape[:2]
        sector_size = width * 5.2
        person_dist = width / 96

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

        if self.show_processed_frame:
            processed_frame = frame.copy()

        for box_index in chosen_boxes:
            index = box_index[0]
            obj_class = self.classes[class_indexes[index]]
            x, y, w, h = boxes[index]
            cx = x + w // 2
            cy = y + h // 2
            end = (x + w, y + h)
            color = (0, 255, 0)
            start = (x - 10, y - 10)

            if self.show_processed_frame:
                processed_frame = cv.rectangle(processed_frame, start, end, color, 2)
                processed_frame = cv.putText(processed_frame, obj_class, start, cv.FONT_HERSHEY_SIMPLEX,
                                             1, color, 2, cv.LINE_AA)

            if self.check_sector(cx, cy, sector_size):
                if len(self.centroids) == 0:
                    self.centroids.append((cx, cy))
                    self.counted[obj_class] += 1
                for i in range(len(self.centroids)):
                    dist = cv.norm(self.centroids[i], (cx, cy), cv.NORM_L2)
                    if dist < person_dist * self.objects_dist[obj_class] * self.dist_coef:
                        self.centroids[i] = (cx, cy)
                        break
                    i += 1
                    if i == len(self.centroids):
                        self.centroids.append((cx, cy))
                        self.counted[obj_class] += 1

        self.logger.info(f'Counted: "{", ".join(self.str_counted())}" on frame {self.frame_id}')

        if self.show_processed_frame:
            text_count = [f"{x}: {i}" for x, i in self.counted.items()]
            text_count = 'Counted: ' + ', '.join(text_count)
            for line in self.lines:
                processed_frame = cv.line(processed_frame, *line, color, 2)
            processed_frame = cv.putText(processed_frame, text_count, (10, height - 10), cv.FONT_HERSHEY_SIMPLEX,
                                         1, color, 2, cv.LINE_AA)
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
            json.dump(data, f)

        self.logger.info(f'Saved "{", ".join(self.str_counted())}" to "{filename}" with id "{camera_id}"')
