import cv2

car_cascade = cv2.CascadeClassifier('./detectors/cascade_config/car.xml')

def get_bounding_boxes(frame):
    bounding_boxes = car_cascade.detectMultiScale(frame, minNeighbors=5)

    classes = None
    confidences = None

    return bounding_boxes, classes, confidences