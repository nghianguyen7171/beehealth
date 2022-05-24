import streamlit as st
import pandas as pd
from PIL import  Image
import Train_window
import Predict_window
import Analysis_window

PAGES = {
    "Data analysis stage": Analysis_window,
    "Training stage": Train_window,
    "Prediction stage": Predict_window
}
logo_path = r'/media/nghia/Nguyen NghiaW/Bee_streamlit/img/Lovepik_com-401013367-lovely-bees.png'
logo = Image.open(logo_path)
resize_logo = logo.resize((100,100))
st.sidebar.image(resize_logo)
st.sidebar.title('STAGE')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()