import time

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
    pred = st.container()

    with header:
        main_logo_path = r'/media/nghia/Nguyen NghiaW/Bee_streamlit/img/favpng_honey-bee-honeycomb-icon.png'
        main_logo = Image.open(main_logo_path).resize((400,400))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(main_logo)
        with col2:
            st.image(main_logo)
        with col3:
            st.image(main_logo)
        st.title('Bee Health Management')
        st.title('PREDICTION')

    col1, col2 = st.columns(2)
    with col1:
        predict = st.button('Predict')
    with col2:
        cancel = st.button('Stop')

    with model:
        logo_path = r'/media/nghia/Nguyen NghiaW/Bee_streamlit/img/Lovepik_com-401013367-lovely-bees.png'
        logo = Image.open(logo_path)
        resize_logo = logo.resize((100, 100))
        st.sidebar.image(resize_logo)
        st.sidebar.selectbox('Select your prediction model:',
                             ('CNN', 'AlexNet', 'LeNet', 'VGG', 'ResNet'))
        thresh = st.sidebar.slider('Select your prediction threshold:', 0.0, 1.0, 0.1)

    with add:
        file = st.file_uploader('Upload your image you want to predict')
        if file is not None:
            #file = r'/media/nghia/Nguyen NghiaW/Bee_streamlit/img/BeeSmall.jpg'
            img = Image.open(file)
            st.title("Here is the image you've selected")
            resized_image = img.resize((336, 336))
            st.image(resized_image)

    st.write('Select your task:')
    subspecies = st.checkbox('Bee subspecies recognition')
    health = st.checkbox('Bee health recognition')

    if subspecies:
        d1 = [['-1', '75%'], ['1 mixed local stock 2', '10%'], ['carniolan honey bee', '7%'], ['Italian honey bee', '5%'],
             ['Russian honey bee', '1.5%'], ['VHS Italian honey bee', '0.5%'],
             ['Western honey bee', '1%']]
        df1 = pd.DataFrame(d1, columns=['Predicted subspecies', 'Confident score'])
    if health:
        d2 = [['ant problem', '75%'], ['few varrao', '10%'], ['healthy', '7%'],
             ['hive being robbed', '5%'], ['missing queen', '1.5%'], ['Varroa, small hive beetles', '0.5%']]
        df2 = pd.DataFrame(d2, columns=['Predicted health', 'Confident score'])


    if predict:
        with st.spinner('Predicting'):
            time.sleep(3)
            if subspecies and health:
                st.write(df1)
                st.write(df2)
            elif subspecies:
                st.write(df1)
            elif health:
                st.write(df2)
            else:
                st.write('**no task is selected!**')
        st.success('Done!')

    if cancel:
        st.write('**Cancel prediction!**')

