import cv2 as cv
import numpy as np


class Counter:
    """Counter object to segment, outline and count objects on camera

    :param yolo_cfg: Path to "yolo3-spp.cfg"
    :type yolo_cfg: str
    :param yolo_weights: Path to "yolo3-spp.weights" from https://pjreddie.com/media/files/yolov3-spp.weights
    :type yolo_weights: str
    :param coco_names: Path to "coco_names"
    :type coco_names: str
    """

    def __init__(self, yolo_cfg, yolo_weights, coco_names):  # TODO: Add coco objects filter
        self.net = cv.dnn.readNet(yolo_cfg, yolo_weights)
        layer_names = self.net.getLayerNames()
        out_layers_indexes = self.net.getUnconnectedOutLayers()
        self.out_layers = [layer_names[index[0] - 1] for index in out_layers_indexes]

        with open(coco_names) as f:
            self.classes = f.read().split('\n')

    def apply_yolo(self, img: np.array):
        """Function to apply YOLOv3 NN to segment COCO objects

        :param img: Image array
        :type img: np.array

        :return boxes_result: List of objects corners
        :rtype boxes_result: list
        :return indexes_result: List of objects IDs in coco.names
        :rtype indexes_result: list
        """
        height, width, _ = img.shape
        blob = cv.dnn.blobFromImage(img, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
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
                if class_score > 0 and class_index < 8:
                    # if class_score>0:
                    cx =  int(obj[0] * width)
                    cy = int(obj[1] * height)
                    obj_width = int(obj[2] * width)
                    obj_height = int(obj[3] * height)
                    x = cx - obj_width // 2
                    y = cy - obj_height // 2
                    box = [x, y, obj_width, obj_height]
                    boxes.append(box)
                    class_indexes.append(class_index)
                    class_scores.append(float(class_score))

        boxes_result, indexes_result = [], []
        chosen_boxes = cv.dnn.NMSBoxes(boxes, class_scores, 0.2, 0.5)
        for i in chosen_boxes:
            boxes_result.append(boxes[i[0]])
            indexes_result.append(class_indexes[i[0]])

        return boxes_result, indexes_result

    def draw_object(self, img, boxes, indexes):
        """Function to draw objects on frame

        :param img: Image array
        :type img: np.array
        :param boxes: List of objects corners from self.apply_yolo
        :type boxes: list
        :param indexes: List of objects IDs in coco.names from self.apply_yolo
        :type indexes: list

        :return img: Image array with outlined objects
        :rtype img: np.array
        """

        for box, index in zip(boxes, indexes):
            x, y, w, h = box
            start = (x, y)
            end = (x + w, y + h)
            color = (255, 255, 255)
            img = cv.rectangle(img, start, end, color, 2)

            start = (x - 10, y - 10)
            text = self.classes[index]
            img = cv.putText(img, text, start, cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)

        return img
