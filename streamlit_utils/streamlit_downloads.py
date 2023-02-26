import base64
import pandas as pd
import io
from zipfile import ZipFile


# Get download link for strings and pandas dataframe
def get_download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

# Get download link for image
def get_image_download_link(img,filename,text):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'


def get_zipped_images_download_link(img_list,filename,text):
    zip_file_bytes_io = io.BytesIO()

    with ZipFile(zip_file_bytes_io, 'w') as zip_file:

        for img_name, img in img_list:
            buffered = io.BytesIO()

            img.save(buffered, format="PNG")
            zip_file.writestr(img_name+".png", buffered.getvalue())

    zip_imgs_str = base64.b64encode(zip_file_bytes_io.getvalue()).decode()
    
    return f'<a href="data:file/txt;base64,{zip_imgs_str}" download="{filename}">{text}</a>'


# Get csv data for download
def get_csv_download_data(object_to_download):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    return object_to_download

# Get image data for download
def get_image_download_data(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")

    return buffered

# Get images zipped data for download
def get_zipped_images_download_data(img_list):
    zip_file_bytes_io = io.BytesIO()

    with ZipFile(zip_file_bytes_io, 'w') as zip_file:
        buffered = io.BytesIO()

        for img_name, img in img_list:
            img.save(buffered, format="PNG")
            zip_file.writestr(img_name+".png", buffered.getvalue())
    
    return zip_file_bytes_io