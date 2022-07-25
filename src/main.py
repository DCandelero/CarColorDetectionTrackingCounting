# Main Libs
import cv2
import numpy as np
import math
import pandas as pd
from PIL import Image
from copy import copy

# My Libs
from kfc_tracker import kfc_tracker
from yolo_detector import yolo_detector
from car_color_classification import prediction
from utils import counter
from utils import visualizer
from utils import utils

# Streamlit Libs
import streamlit as st
from streamlit_utils import streamlit_downloads

def main():
    cap = cv2.VideoCapture('../Data/Traffic_Example.mp4')

    car_cascade = cv2.CascadeClassifier('./car.xml')


    # Streamlit page config
    st.set_page_config(page_title='Car&Color Detection', page_icon=':car', layout="wide")
    st_cols = st.columns([8,1,1,1,1,1])
    st_video = st_cols[0].empty()
    st_df = st_cols[0].empty()
    # Streamlit sidebar config
    # Sidebar Control
    st_control_option = st.sidebar.selectbox('Control Comands', ['Stop', 'Start', 'Restart'])
    # Sidebar Upload
    st_upload_option = st.sidebar.selectbox('Choose your upload option', ['Sample video', 'Youtube link', 'File upload'])
    if(st_upload_option == 'Youtube link'):
        st_upload_option_value = st.sidebar.text_input('Paste youtube link here!')
    elif(st_upload_option == 'File upload'):
        st_upload_option_value = st.sidebar.file_uploader('Upload your mp4 file here!')
    else:
        st.sidebar.write('Sample video is being used')
    # Sidebar Download
    st_download_flag= st.sidebar.select_slider('Enable data download', options=['No', 'Yes'])
    if(st_download_flag == 'Yes'):
        st_csv_download_option = st.sidebar.checkbox('CSV DataFrame')
        st_zip_images_download_option = st.sidebar.checkbox('Last car 30 images (PNG)') 
    st_df_download = st.sidebar.empty() # DataFrame Download
    st_zip_download = st.sidebar.empty() # Zipped images Download

    # Pandas Dataframe
    df_cars = pd.DataFrame(columns=['CarID', 'CountOrder', 'PredictedColor', 'VideoFrame', 'BoundingBox'])

    # Variables ------------------------------------------------
    cars_already_counted = 0
    cars_already_counted_id = []
    cars_already_counted_imgs = [] # [(img_name(str), car_img(np_array))]
    timer = cv2.getTickCount()

    # Detector definitions -------------------------------------
    blobs = {}
    max_consecutive_failures = 2
    detection_interval = 4

    ret, frame = cap.read()
    frame = cv2.resize(frame, (960, 540))

    counting_lines = [
        [(0, int(frame.shape[0]/2)), (frame.shape[1], int(frame.shape[0]/2))]
    ]
    counts = 0
    # Output config
    output_video = cv2.VideoWriter("../Output/objectCounter.avi",
        cv2.VideoWriter_fourcc(*'MJPG'), 
        30,
        (frame.shape[1], frame.shape[0])
    )

    # Start detection
    _bounding_boxes, _classes, _confidences = yolo_detector.get_bounding_boxes(frame)
    blobs = kfc_tracker.add_new_blobs(_bounding_boxes, _classes, _confidences, blobs, frame, max_consecutive_failures)

    frame_count = 0
    while(ret and st_control_option != 'Stop'):

        if(frame is None):
            print("Frame vazio")
            print(cap.read())

        frame = cv2.resize(frame, (960, 540))

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

        # Save car imgs, predict car colors and add elements to streamlit
        if(counts > cars_already_counted):
            
            # Get cars that was counted only in this frame (car_id, car_img)
            cars = utils.get_car_counted(blobs_list, cars_already_counted_id, frame)

            car_color_predictions = prediction.get_car_color_predictions(cars)

            for idx, (car_id, car_img, color_predicted) in enumerate(car_color_predictions):
                cars_already_counted_imgs.append((color_predicted+car_id, car_img))

                # Padding images to align width and height of the columns
                car_img_pad = utils.pad_images(car_img, 200)

                # Update dataframe
                df_car = pd.DataFrame([[car_id, cars_already_counted+idx, color_predicted, '-', '-']], columns=['CarID', 'CountOrder', 'PredictedColor', 'VideoFrame', 'BoundingBox'])
                df_cars = df_cars.append(df_car, ignore_index=True)

                # Display imgs and df on streamlit
                st_df.dataframe(df_cars)
                st_colum_idx = cars_already_counted%5 + 1 + idx
                st_cols[st_colum_idx].image(car_img_pad)
                st_cols[st_colum_idx].write(color_predicted)
                
                # Update download button on streamlit
                if(st_download_flag == 'Yes'):
                    if(st_csv_download_option):
                        tmp_download_link = streamlit_downloads.get_download_link(df_cars, 'df_cars.csv', 'Click here to download your data!')
                        st_df_download.markdown(tmp_download_link, unsafe_allow_html=True)
                    if(st_zip_images_download_option):
                        cars_already_counted_imgs_pil  = [(img_name, Image.fromarray(np_img)) for (img_name, np_img) in cars_already_counted_imgs]
                        st_sidebar2.markdown(streamlit_downloads.get_zipped_images_download_link(cars_already_counted_imgs_pil, 'cars_imgs.zip', 'Download ZIP'), unsafe_allow_html=True)


            cars_already_counted = counts


        # Calculate fps
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        timer = cv2.getTickCount()

        # Create outputframe
        frame_copy = copy(frame)
        output_frame = visualizer.visualize(frame_copy, blobs, fps, counts)

        # Record frames
        output_video.write(output_frame)

        # Display frame (Streamlit)
        st_video.image(output_frame)

        # Display result ----------------------------------------------------------------------------
        cv2.imshow("output_frame", output_frame)
        frame_count += 1
        ret, frame = cap.read()

        if cv2.waitKey(25) & 0xFF == ord('q') or st_control_option == 'Stop':
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()