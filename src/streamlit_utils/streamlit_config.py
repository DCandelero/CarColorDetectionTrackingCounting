import streamlit as st
from streamlit_utils import streamlit_downloads
import pafy

def set_streamlit_layout():
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
        # if st_upload_option_value != '':
        #     video = pafy.new(st_upload_option_value)
        #     st_upload_option_value = video.getbest(preftype="mp4")
    elif(st_upload_option == 'File upload'):
        st_upload_option_value = st.sidebar.file_uploader('Upload your mp4 file here!', type=['mp4'], help='Available extensions: Only .mp4')
        if st_upload_option_value is not None:
            data_file = open('../Data/uploaded_file.mp4', 'wb')
            data_file.write(st_upload_option_value.getvalue())
            data_file.close()
            st_upload_option_value = '../Data/uploaded_file.mp4'
        else:
            st_upload_option_value = '../Data/Traffic_Example.mp4'
    else:
        radio_choice = st.sidebar.radio('Choose which sample video will be used:', ['Exemplo1', 'Exemplo2'])
        if(radio_choice == 'Exemplo1'):
            st_upload_option_value = '../Data/Traffic_Example.mp4'
        else:
            st_upload_option_value = '../Data/videoplayback.webm'
    # Sidebar Download
    st_download_flag= st.sidebar.select_slider('Enable data download', options=['No', 'Yes'])
    if(st_download_flag == 'Yes'):
        st_csv_download_option = st.sidebar.checkbox('CSV DataFrame')
        st_zip_images_download_option = st.sidebar.checkbox('Download car images (PNG)')
    else:
        st_csv_download_option = 0
        st_zip_images_download_option = 0
    st_df_download = st.sidebar.empty() # DataFrame Download
    st_zip_download = st.sidebar.empty() # Zipped images Download

    if(st_control_option == 'Stop'):
        with open('../Output/objectCounter.avi', 'rb') as f:
            st.sidebar.download_button('Download video', f, file_name='output.mp4')

    return (st_cols, st_video, st_df, st_control_option, st_download_flag, st_csv_download_option, 
        st_zip_images_download_option, st_df_download, st_zip_download, st_upload_option_value)


# # remove hamburger menu top left side
# st.markdown(""" <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style> """, unsafe_allow_html=True)


# # remove layout padding
# padding = 0
# st.markdown(f""" <style>
#     .reportview-container .main .block-container{{
#         padding-top: {padding}rem;
#         padding-right: {padding}rem;
#         padding-left: {padding}rem;
#         padding-bottom: {padding}rem;
#     }} </style> """, unsafe_allow_html=True)


# https://blog.streamlit.io/introducing-theming/
# https://docs.streamlit.io/library/advanced-features/configuration
# create a .streamlit folder in my repository and add this config.toml file
# # Custom color pallete
# [theme]
# primaryColor="#2214c7"
# backgroundColor="#ffffff"
# secondaryBackgroundColor="#e8eef9"
# textColor="#000000"
# font="sans serif"