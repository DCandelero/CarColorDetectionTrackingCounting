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