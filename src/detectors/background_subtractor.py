import cv2

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=150)

def get_bounding_boxes(frame):
    bounding_boxes = []
    # Threshold to draw the rectangle (bounding box)
    area_threshold = (frame.shape[0]*frame.shape[1])*0.003 #0.3% of the image

    # Create mask (Detects objects that are in motion)
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) #Remove shadow

    # Get contours based on mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)

        # Get bounding box
        if (area > area_threshold):
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

    classes = None
    confidences = None

    return bounding_boxes, classes, confidences