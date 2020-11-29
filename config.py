from typing import Tuple, List

CAMERA_ID: str or int = "test"
POINTS: list or tuple = (((630, 520), (770, 525)), ((850, 610), (1890, 480)))
CLASSES: List[str] or Tuple[str] = ("person", "car", "bus", "bicycle", "motorbike", "truck")

YOLO_FILES_DIR: str = "yolo_files"

VIDEO_PATH: str = "test_vid.avi"
OUTPUT_VIDEO_PATH: str = "output.mp4"  # Output video supports only MP4
OUTPUT_JSON_PATH: str = "output.json"
