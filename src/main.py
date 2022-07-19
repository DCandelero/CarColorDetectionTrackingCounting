import cv2
import numpy as np
import math

from kfc_tracker import kfc_tracker
from yolo_detector import yolo_detector
from car_color_classification import prediction
from utils import counter
from utils import visualizer
from utils import utils


def main():
    cap = cv2.VideoCapture('../Data/Traffic_Example.mp4')
    print(cap.isOpened())

    car_cascade = cv2.CascadeClassifier('./car.xml')

    cars_counted = []
    cars_already_counted = 0

    blobs = {}
    max_consecutive_failures = 2
    detection_interval = 5

    ret, frame = cap.read()

    frame = cv2.resize(frame, (960, 540))

    frame_shape = frame.shape
    print(frame_shape)
    counting_lines = [
        [(0, int(frame_shape[0]/2)), (frame_shape[1], int(frame_shape[0]/2))]
    ]
    counts = 0

    output_video = cv2.VideoWriter("../Output/objectCounter.avi",
        cv2.VideoWriter_fourcc(*'MJPG'), 
        30,
        (frame.shape[1], frame.shape[0])
    )

    _bounding_boxes, _classes, _confidences = yolo_detector.get_bounding_boxes(frame)
    blobs = kfc_tracker.add_new_blobs(_bounding_boxes, _classes, _confidences, blobs, frame, max_consecutive_failures)

    frame_count = 0
    while(cap.isOpened()):

        frame = cv2.resize(frame, (960, 540))

        # Start timer
        timer = cv2.getTickCount()
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # update blob trackers
        blobs_list = list(blobs.items())
        blobs_list = [kfc_tracker.update_blob_tracker(blob, blob_id, frame) for blob_id, blob in blobs_list]
        blobs = dict(blobs_list)

        # Count vehicles
        for blob_id, blob in blobs_list:
            # count object if it has crossed a counting line
            blob, counts = counter.attempt_count(blob, counting_lines, counts)
            blobs[blob_id] = blob
            # remove blob if it has reached the limit for tracking failures
            if blob.num_consecutive_tracking_failures >= max_consecutive_failures:
                del blobs[blob_id]

        # detect objects
        if frame_count % detection_interval == 0:
            # rerun detection
            _bounding_boxes, _classes, _confidences = yolo_detector.get_bounding_boxes(frame)

            blobs = kfc_tracker.add_new_blobs(
                _bounding_boxes, _classes, _confidences, 
                blobs, frame, max_consecutive_failures)
            blobs = kfc_tracker.remove_duplicates(blobs)

        # Print blobs
        if(counts > cars_already_counted):
            cars_already_counted = counts

            cars = utils.get_car_counted(blobs_list, cars_counted, frame)

            car_color_predictions = prediction.get_car_color_predictions(cars)

            for (car_id, car_img, color_predicted) in car_color_predictions:
                cv2.imwrite('../CarsCounted/'+color_predicted+'_'+car_id+'.png', car_img)

        output_frame = visualizer.visualize(frame, blobs, fps, counts)

        # Record frames
        output_video.write(output_frame)

        # Display result ----------------------------------------------------------------------------
        cv2.imshow("output_frame", output_frame)
        frame_count += 1
        ret, frame = cap.read()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()