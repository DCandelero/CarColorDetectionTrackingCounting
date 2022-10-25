import os
import cv2
import numpy as np
import gdown
import zipfile

def download_data(url, folder_path_to_save):
    data_path = os.path.join(folder_path_to_save, 'yolov3.weights')
    # Donwload zip
    gdown.download(url, data_path, quiet=False, fuzzy=True)


yolov3_config_folder_path = os.path.join(os.path.dirname(__file__), 'yolo_config')
yolov3_weights_path = os.path.join(yolov3_config_folder_path, 'yolov3.weights')
if(not os.path.isfile(yolov3_weights_path)):
    download_data('https://drive.google.com/file/d/1unjeL1KZCpUqeDwKCPeR53OCG9A_7x2d/view?usp=sharing', yolov3_config_folder_path)


with open(os.path.join(yolov3_config_folder_path, 'coco_classes.txt'), 'r') as classes_file:
    CLASSES = dict(enumerate([line.strip() for line in classes_file.readlines()]))
with open(os.path.join(yolov3_config_folder_path, 'coco_classes_of_interest.txt'), 'r') as coi_file:
    CLASSES_OF_INTEREST = tuple([line.strip() for line in coi_file.readlines()])
conf_threshold = 0.5
net = cv2.dnn.readNet(yolov3_weights_path, os.path.join(yolov3_config_folder_path, 'yolov3.cfg'))

def get_bounding_boxes(image):
    '''
    Return a list of bounding boxes of objects detected,
    their classes and the confidences of the detections made.
    '''

    # create image blob
    scale = 0.00392
    image_blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    # detect objects
    net.setInput(image_blob)
    layer_names = net.getLayerNames()
    output_layers = []
    for i in net.getUnconnectedOutLayers():
        output_layers.append(layer_names[i - 1])
    outputs = net.forward(output_layers)

    classes = []
    confidences = []
    boxes = []
    nms_threshold = 0.4

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and CLASSES[class_id] in CLASSES_OF_INTEREST:
                width = image.shape[1]
                height = image.shape[0]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                classes.append(CLASSES[class_id])
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)


    _bounding_boxes = []
    _classes = []
    _confidences = []
    for i in indices:
        _bounding_boxes.append(boxes[i])
        _classes.append(classes[i])
        _confidences.append(confidences[i])

    return _bounding_boxes, _classes, _confidences