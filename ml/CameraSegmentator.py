from .Mask_RCNN import coco
from .Mask_RCNN import model
from .Mask_RCNN import visualize


# Create configuration
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# Create main class
class CameraSegmentator:
    def __init__(self, coco_weights_path="weights/mask_rcnn_coco.h5", config=InferenceConfig()):
        """ Initialize main parameters.
        Args:
            coco_weights_path (str): Path to coco weights file (Download from internet).
        """
        self.coco_weights_path = coco_weights_path

        self.model_train, self.model_predict = None, None

        self.model_dir = "checkpoints"
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                            'bus', 'train', 'truck', 'boat', 'traffic light',
                            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                            'kite', 'baseball bat', 'baseball glove', 'skateboard',
                            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                            'teddy bear', 'hair drier', 'toothbrush']

        self.config = config

    def _init_predict(self):
        self.model_predict = model.MaskRCNN(mode="inference", model_dir=self.model_dir, config=self.config)
        self.model_predict.load_weights(self.coco_weights_path, by_name=True)

    def predict(self, images):  # TODO: Fix images different sizes
        """ Predict a set of images using the predict model
        Args:
            images: List with image(s) to predict.
        Returns:

        """
        self._init_predict()
        length = len(images)
        predictions = []

        index = 0
        while index < length:
            print(index, self.config.BATCH_SIZE, length)
            predictions.append(self.model_predict.detect(images[index:index + self.config.BATCH_SIZE]))
            index += self.config.BATCH_SIZE

        print(len(images))
        print(index)
        delta = len(images) - index
        if delta:
            images += images[:delta]
            predictions.append(self.model_predict.detect(images[index:index + self.config.BATCH_SIZE]))
            return predictions[:-delta]
        else:
            return predictions

    def predict_video(self, video, frame_delta):  # TODO
        self._init_predict()

    def visualize(self, images, preds):
        """Visualize prediction to image with masks and labels.

        Args:
            images: List of image(s).
            preds: List of result(s) from predict function.
        Returns:
            Images with visualized predictions.
        """
        visualized = []
        for image, pred in zip(images, preds):
            visualized.append(visualize.display_instances(image, pred[0]['rois'], pred[0]['masks'],
                                                          pred[0]['class_ids'], self.class_names, pred[0]['scores']))
        return visualized
