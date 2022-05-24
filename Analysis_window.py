import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from PIL import  Image
import time
from plotly.offline import iplot

def app():
    header = st.container()
    nav = st.container()
    add = st.container()
    vis = st.container()

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
        st.title('DATA ANALYSIS')

    with nav:
        logo_path = r'/media/nghia/Nguyen NghiaW/Bee_streamlit/img/Lovepik_com-401013367-lovely-bees.png'
        logo = Image.open(logo_path)
        resize_logo = logo.resize((100, 100))
        st.sidebar.image(resize_logo)
        st.sidebar.write('Select visualization method:')
        vis_sub = st.sidebar.checkbox('Sample bee images by subspecices')
        vis_health = st.sidebar.checkbox('Sample bee images by health')
        subspec = st.sidebar.checkbox('Subspecies distribution')
        health_dis = st.sidebar.checkbox('Health distribution')
        locate_dis = st.sidebar.checkbox('Location distribution')
        img_loca_heal_sub = st.sidebar.checkbox('Number of image per location, health  and subspecies')
        img_time = st.sidebar.checkbox('Number of samples collected by time')
        corr = st.sidebar.checkbox('Correlation map')

    with add:
        tab = st.file_uploader('Upload your dataset')  # , type=['csv', 'xlsx']
        if tab:
            tab = r'/home/nghia/CNU-SEM3/AI_proj/Bee_Health/Bee/bee_imgs/bee_data.csv'
            df = pd.read_csv(tab)
            st.write(df.head(5))

    with vis:
        st.write('Data visualization')
        if vis_sub:
            st.image(Image.open(r'/home/nghia/Downloads/Bee/subspecies_s.png'))
        if vis_health:
            st.image(Image.open(r'/home/nghia/Downloads/Bee/healthy_bee.png'))
            st.image(Image.open(r'/home/nghia/Downloads/Bee/sick_bee.png'))
        if subspec:
            st.image(Image.open(r'/home/nghia/Downloads/Bee/subspecies.png'))
        if health_dis:
            st.image(Image.open(r'/home/nghia/Downloads/Bee/health_dis.png'))
        if locate_dis:
            st.image(Image.open(r'/home/nghia/Downloads/Bee/Bee_location.png'))
        if img_loca_heal_sub:
            st.image(Image.open(r'/home/nghia/Downloads/Bee/num_img_per_loca_health_sub.png'))
        if img_time:
            st.image(Image.open(r'/home/nghia/Downloads/Bee/Bee_per_day.png'))
        if corr:
            st.image(Image.open(r'/home/nghia/Downloads/Bee/num_img_per_sub_health.png'))
            st.image(Image.open(r'/home/nghia/Downloads/Bee/subspec_hour.png'))
            st.image(Image.open(r'/home/nghia/Downloads/Bee/subspec_loca.png'))
