import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from PIL import  Image
from plotly.offline import iplot

def app():
    header = st.container()
    model = st.container()
    add = st.container()
    train = st.container()

    with header:
        main_logo_path = r'/media/nghia/Nguyen NghiaW/Bee_streamlit/img/favpng_honey-bee-honeycomb-icon.png'
        main_logo = Image.open(main_logo_path).resize((400, 400))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(main_logo)
        with col2:
            st.image(main_logo)
        with col3:
            st.image(main_logo)
        st.title('Bee Health Management')
        st.title('TRAINING')

    col1, col2 = st.columns(2)
    with col1:
        train_progress = st.button('Train')
    with col2:
        cancel = st.button('Cancel')
    #with col3:
     #   jump = st.button('Go to prediction window')

    if train_progress:
        st.write('Training...')
    if cancel:
        st.write('Cancel training!')

    with model:
        logo_path = r'/media/nghia/Nguyen NghiaW/Bee_streamlit/img/Lovepik_com-401013367-lovely-bees.png'
        logo = Image.open(logo_path)
        resize_logo = logo.resize((100,100))
        st.sidebar.image(resize_logo)
        st.sidebar.selectbox('Select your training model:',
                             ('CNN', 'AlexNet', 'LeNet'))
        thresh = st.sidebar.slider('Select your threshold:', 0.0, 1.0, 0.1)

    with add:
        file = st.file_uploader('Upload your image')
        if file:
            file = r'/media/nghia/Nguyen NghiaW/Bee_streamlit/img/BeeSmall.jpg'
            img = Image.open(file)
            st.title("Here is the image you've selected")
            resized_image = img.resize((336, 336))
            st.image(resized_image)

        tab = st.file_uploader('Upload your dataset') #, type=['csv', 'xlsx']
        if tab:
            tab = r'/home/nghia/CNU-SEM3/AI_proj/Bee_Health/Bee/bee_imgs/bee_data.csv'
            df = pd.read_csv(tab)
            st.write(df.head(5))

    with train:
        ratio = st.slider('Select training ratio (%):')
        st.write('training rate is ', ratio,'%')
