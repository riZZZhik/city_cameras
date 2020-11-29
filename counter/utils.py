import os
import cv2 as cv


def wget_file(filepath: str, file_url: str):
    if not os.path.exists(filepath):
        print(f'{filepath} file not found, downloading from server')
        shell_script = f"wget -O {filepath} {file_url}"

        if os.system(shell_script):
            print("Root permission required, trying with sudo")
            if os.system("sudo " + shell_script):
                raise Exception(f"Unable to download {filepath}, connect a specialist")

    return filepath


def get_yolo_files(yolo_dir, yolo_paths):
    # Assertions
    assert yolo_dir or yolo_paths, "You should enter yolo_dir or yolo_paths"

    if yolo_dir:
        if not os.path.exists(yolo_dir):
            os.mkdir(yolo_dir)
    else:
        yolo_files = ("coco.names", "yolov3-spp.cfg", "yolov3-spp.weights")
        assert all((x in yolo_paths for x in yolo_files)), f"yolo_paths should have {', '.join(yolo_files)} files"

    # Get files paths
    try:
        coco_names = yolo_paths["coco.names"]
    except KeyError:
        coco_names = os.path.join(yolo_dir, "coco.names")

    try:
        yolo_cfg = yolo_paths["yolov3-spp.cfg"]
    except KeyError:
        yolo_cfg = os.path.join(yolo_dir, "yolov3-spp.cfg")

    try:
        yolo_weights = yolo_paths["yolov3-spp.weights"]
    except KeyError:
        yolo_weights = os.path.join(yolo_dir, "yolov3-spp.weights")

    # Download files if it is needed
    wget_file(coco_names, "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")
    wget_file(yolo_cfg, "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-spp.cfg")
    wget_file(yolo_weights, "https://pjreddie.com/media/files/yolov3-spp.weights")

    return coco_names, yolo_cfg, yolo_weights


def init_yolo(yolo_dir, yolo_paths):
    coco_names, yolo_cfg, yolo_weights = get_yolo_files(yolo_dir, yolo_paths)

    # Split coco_names
    with open(coco_names) as f:
        coco_classes = {i: x for i, x in enumerate(f.read().split('\n'))}

    # Init YOLO
    net = cv.dnn.readNet(yolo_cfg, yolo_weights)
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index[0] - 1] for index in out_layers_indexes]

    return coco_classes, net, out_layers
