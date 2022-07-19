import cv2

font = cv2.FONT_HERSHEY_DUPLEX
line_type = cv2.LINE_AA

def visualize(frame, blobs, fps, counts):
    # Display tracker and fps
    cv2.putText(frame, "KCF Tracker", (int(frame.shape[1]*0.05) ,int(frame.shape[0]*0.1)), font, 0.75, (50,170,50),2) 
    cv2.putText(frame, "FPS : " + str(int(fps)), (int(frame.shape[1]*0.05),int(frame.shape[0]*0.15)), font, 0.75, (50,170,50), 2)
    
    # draw and label blob bounding boxes
    for _id, blob in blobs.items():
        (x, y, w, h) = [int(v) for v in blob.bounding_box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        object_label = 'I: ' + _id[:8] \
            if blob.type is None \
            else 'I: {0}, T: {1} ({2})'.format(_id[:8], blob.type, str(blob.type_confidence)[:4])
        cv2.putText(frame, object_label, (x, y - 5), font, 1, (255, 0, 0), 2, line_type)

    # Draw counting line
    line_thickness = 2
    cv2.line(frame, (0, int(frame.shape[0]/2)), (frame.shape[1], int(frame.shape[0]/2)), (0, 255, 0), thickness=line_thickness)

    # Draw
    cv2.putText(frame, "Count: {}".format(counts), (int(frame.shape[1]*0.8), int(frame.shape[0]*0.1)), font, 1, (0, 0, 255), 2, line_type)
    
    
    return frame