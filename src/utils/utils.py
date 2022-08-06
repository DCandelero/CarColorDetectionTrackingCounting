import numpy as np

def get_car_counted(blobs_list, cars_counted, frame):
    cars = [] # (car_id, car_img) 

    for blob in blobs_list:
        blob_id = blob[0]
        blob_obj = blob[1]
        
        if(blob_id not in cars_counted and blob_obj.crossed_counting_line):
            cars_counted.append(blob_id)

            x = int(blob_obj.bounding_box[0] - (blob_obj.bounding_box[0]+blob_obj.bounding_box[2])*0.04)
            y = int(blob_obj.bounding_box[1] - (blob_obj.bounding_box[1]+blob_obj.bounding_box[3])*0.08)
            w = int(blob_obj.bounding_box[0]+blob_obj.bounding_box[2] + (blob_obj.bounding_box[0]+blob_obj.bounding_box[2])*0.04)
            h = int(blob_obj.bounding_box[1]+blob_obj.bounding_box[3] + (blob_obj.bounding_box[1]+blob_obj.bounding_box[3])*0.08)
            car_img = frame[y:h, x:w]

            cars.append((blob_id, car_img))
            
    return cars

def pad_images(img, padding_limit):
    if(img.shape[0] < padding_limit):
        top_padding = int((padding_limit - img.shape[0]) / 2)
        bottom_padding = int((padding_limit - img.shape[0]) / 2)
    else:
        top_padding = 0
        bottom_padding = 0
    if(img.shape[1] < padding_limit):
        left_padding = int((padding_limit - img.shape[1]) / 2)
        right_padding = int((padding_limit - img.shape[1]) / 2)
    else:
        left_padding = 0
        right_padding = 0
    pad_image = np.pad(img, ((top_padding, bottom_padding), (left_padding, right_padding), (0, 0)), constant_values=255)[:,:,0]

    return pad_image