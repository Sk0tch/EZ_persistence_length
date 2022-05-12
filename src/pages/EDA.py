import os
import streamlit as st
import numpy as np
import pandas as pd

import plotly.express as px
from src.pages.Utils import LoadData, Parser, LinearMath, Painter, DataProcessor

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def EDA():
    st.header('Exploratory Data Analysis')

    data_coord = LoadData('coord')
    data_angles = LoadData('angles')
    group = LoadData('group')
    max_chains = data_coord['chain_ind'].max()
    st.write('chains: ', max_chains)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('coord data:')
        st.dataframe(data_coord.head(5))
        csv_coord = convert_df(data_coord)
        st.download_button(
             label="Download data as CSV",
             data=csv_coord,
             file_name='coord_data.csv',
             mime='text/csv',
         )
    with col2:
        st.write('angles data:')
        st.dataframe(data_angles.head(5))
        csv_angles = convert_df(data_angles)
        st.download_button(
             label="Download data as CSV",
             data=csv_angles,
             file_name='data_angles.csv',
             mime='text/csv',
         )
    with col3:
        st.write('grouping data:')
        st.dataframe(group.head(5))
        csv_group = convert_df(group)
        st.download_button(
             label="Download data as CSV",
             data=csv_group,
             file_name='group_data.csv',
             mime='text/csv',
         )


    # st.dataframe(data.head(5))
    chains_numb = st.number_input(label='number of chains to display', min_value=1, max_value=max_chains, step=1)
    st.plotly_chart(Painter.plot_chains(data_coord, chains_numb))
