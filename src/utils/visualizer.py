import cv2

font = cv2.FONT_HERSHEY_DUPLEX
line_type = cv2.LINE_AA

def visualize(frame, blobs, fps, counts):
    # Display tracker and fps
    cv2.putText(frame, "KCF Tracker", (int(frame.shape[1]*0.05) ,int(frame.shape[0]*0.9)), font, 0.75, (255,255,255),1) 
    cv2.putText(frame, "FPS : " + str(int(fps)), (int(frame.shape[1]*0.05),int(frame.shape[0]*0.95)), font, 0.75, (255,255,255), 1)
    
    # draw and label blob bounding boxes
    for _id, blob in blobs.items():
        (x, y, w, h) = [int(v) for v in blob.bounding_box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        object_label = 'I: ' + _id[:8] \
            if blob.type is None \
            else 'I: {0}, T: {1} ({2})'.format(_id[:8], blob.type, str(blob.type_confidence)[:4])
        cv2.putText(frame, object_label, (x, y - 5), font, 0.5, (0, 0, 255), 1, line_type)

    # Draw counting line
    line_thickness = 2
    cv2.line(frame, (0, int(frame.shape[0]/2)), (frame.shape[1], int(frame.shape[0]/2)), (0, 255, 0), thickness=line_thickness)

    # Draw counter
    cv2.putText(frame, "Count: {}".format(counts), (int(frame.shape[1]*0.8), int(frame.shape[0]*0.9)), font, 0.75, (255, 255, 255), 1, line_type)


    cv2.line(frame, (0, int(frame.shape[0]*0.8)), (frame.shape[1], int(frame.shape[0]*0.8)), (0, 0, 0), thickness=1)

    # Draw transparency factor
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, int(frame.shape[0]*0.8)), (frame.shape[1], frame.shape[0]), (0, 0, 0), thickness=-1)
    alpha = 0.4  # Transparency factor.
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame