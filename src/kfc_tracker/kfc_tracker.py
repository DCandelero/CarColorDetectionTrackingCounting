import cv2
import uuid

def add_new_blobs(boxes, classes, confidences, blobs, frame, mcdf):
    matched_blob_ids = []
    for i, box in enumerate(boxes):
        _type = classes[i] if classes is not None else None
        _confidence = confidences[i] if confidences is not None else None
        _tracker =  cv2.TrackerKCF_create()
        _tracker.init(frame, tuple(box))

        match_found = False
        for _id, blob in blobs.items():
            if get_overlap(box, blob.bounding_box) >= 0.6:
                match_found = True
                if _id not in matched_blob_ids:
                    blob.num_consecutive_detection_failures = 0
                    matched_blob_ids.append(_id)
                blob.update(box, _type, _confidence, _tracker)


        if not match_found:
            _blob = Blob(box, _type, _confidence, _tracker)
            blob_id = generate_object_id()
            blobs[blob_id] = _blob

    blobs = _remove_stray_blobs(blobs, matched_blob_ids, mcdf)
    return blobs

def remove_duplicates(blobs):
    for blob_id, blob_a in list(blobs.items()):
        for _, blob_b in list(blobs.items()):
            if blob_a == blob_b:
                break

            if get_overlap(blob_a.bounding_box, blob_b.bounding_box) >= 0.6 and blob_id in blobs:
                del blobs[blob_id]
    return blobs

def update_blob_tracker(blob, blob_id, frame):
    '''
    Update a blob's tracker object.
    '''
    success, box = blob.tracker.update(frame)
    if success:
        blob.num_consecutive_tracking_failures = 0
        blob.update(box)
    else:
        blob.num_consecutive_tracking_failures += 1

    return (blob_id, blob)

def _remove_stray_blobs(blobs, matched_blob_ids, mcdf):
    '''
    Remove blobs that "hang" after a tracked object has left the frame.
    '''
    for blob_id, blob in list(blobs.items()):
        if blob_id not in matched_blob_ids:
            blob.num_consecutive_detection_failures += 1
        if blob.num_consecutive_detection_failures > mcdf:
            del blobs[blob_id]
    return blobs

class Blob:
    '''
    A blob represents a tracked object as it moves around in a video.
    '''
    def __init__(self, _bounding_box, _type, _confidence, _tracker):
        self.bounding_box = _bounding_box
        self.type = _type
        self.type_confidence = _confidence
        self.centroid = get_centroid(_bounding_box)
        self.area = get_area(_bounding_box)
        self.tracker = _tracker
        self.num_consecutive_tracking_failures = 0
        self.num_consecutive_detection_failures = 0
        self.crossed_counting_line = False
        self.position_first_detected = tuple(self.centroid)

    def update(self, _bounding_box, _type=None, _confidence=None, _tracker=None):
        self.bounding_box = _bounding_box
        self.type = _type if _type is not None else self.type
        self.type_confidence = _confidence if _confidence is not None else self.type_confidence
        self.centroid = get_centroid(_bounding_box)
        self.area = get_area(_bounding_box)
        if _tracker:
            self.tracker = _tracker


'''
# Bounding box utility functions.
'''

def get_centroid(bbox):
    # Calculates the center point of a bounding box.
    x, y, w, h = bbox
    return (round((x + x + w) / 2), round((y + y + h) / 2))

def get_area(bbox):
    '''
    Calculates the area of a bounding box.
    '''
    _, _, w, h = bbox
    return w * h

def get_overlap(bbox1, bbox2):
    '''
    Calculates the degree of overlap of two bounding boxes.
    This can be any value from 0 to 1 where 0 means no overlap and 1 means complete overlap.
    The degree of overlap is the ratio of the area of overlap of two boxes and the area of the smaller box.
    '''
    bbox1_x1 = bbox1[0]
    bbox1_y1 = bbox1[1]
    bbox1_x2 = bbox1[0] + bbox1[2]
    bbox1_y2 = bbox1[1] + bbox1[3]

    bbox2_x1 = bbox2[0]
    bbox2_y1 = bbox2[1]
    bbox2_x2 = bbox2[0] + bbox2[2]
    bbox2_y2 = bbox2[1] + bbox2[3]

    overlap_x1 = max(bbox1_x1, bbox2_x1)
    overlap_y1 = max(bbox1_y1, bbox2_y1)
    overlap_x2 = min(bbox1_x2, bbox2_x2)
    overlap_y2 = min(bbox1_y2, bbox2_y2)

    overlap_width = overlap_x2 - overlap_x1
    overlap_height = overlap_y2 - overlap_y1

    if overlap_width < 0 or overlap_height < 0:
        return 0.0

    overlap_area = overlap_width * overlap_height

    bbox1_area = (bbox1_x2 - bbox1_x1) * (bbox1_y2 - bbox1_y1)
    bbox2_area = (bbox2_x2 - bbox2_x1) * (bbox2_y2 - bbox2_y1)
    smaller_area = bbox1_area if bbox1_area < bbox2_area else bbox2_area

    epsilon = 1e-5 # small value to prevent division by zero
    overlap = overlap_area / (smaller_area + epsilon)
    return overlap

def generate_object_id():
    return 'obj_' + uuid.uuid4().hex