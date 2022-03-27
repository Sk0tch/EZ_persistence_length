import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os as os
from urllib.request import urlopen
import json

from Utils import Parser, LinearMath, Painter
# import Utils as Utils

def DataInput():
    st.subheader('upload your data')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.read().decode("utf-8") 
    return bytes_data



# Настройка заголовка и текста 
st.title("Polymer chains analys")
# st.write("""This dashboard will present the spread of COVID-19 in the world by visualizing the timeline of the total cases and deaths. As well as the total number of vaccinated people.""")

# Настройка боковой панели
st.sidebar.title("About")
st.sidebar.info(
    """
    This app is Open Source dashboard.
    """
)



# data_folder = 'Data'
# file_name = '2019.04.08 PUC19_2019.04.01 PUC19_mica.dat'
# file_path = os.path.join(data_folder, file_name)

# file_path = DataInput()
# st.write(DataInput())

multy_chain = Parser.parse_data(DataInput())
df_work = Parser.multichain_list_to_frame(multy_chain, calc='angle')
df_work['distance'] = df_work['distance'].round(-1)
group = df_work.groupby('distance').agg({'angle':['mean', 'count']}).reset_index()
group.columns = ['distance', 'ang_mean', 'count']
print('finish')


show_data = st.sidebar.checkbox('Show raw data')
if show_data == True:
    st.subheader('Raw data')
    st.write(df_work.head())


# Создадим поле выбора для визуализации общего количества случаев, смертей или вакцинаций
select_event = st.sidebar.selectbox('Show chart', ('Plot chains', 'Plot angle chart', 'total_vaccinations'))
if select_event == 'Plot chains':
    st.write(Painter.plot_chain(multy_chain, max_chains=10))

if select_event == 'Plot angle chart':
    x = group['distance']
    y = group['ang_mean']
    color = group['count']
    st.plotly_chart(Painter.plot_line_color(x, y, color))

if select_event == 'total_vaccinations':
    st.write('Oooops')