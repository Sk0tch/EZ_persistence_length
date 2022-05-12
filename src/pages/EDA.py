import os
import streamlit as st
import numpy as np
import pandas as pd

import plotly.express as px
from src.pages.Utils import LoadData, Parser, LinearMath, Painter, DataProcessor



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
    with col2:
        st.write('angles data:')
        st.dataframe(data_angles.head(5))
    with col3:
        st.write('grouping data:')
        st.dataframe(group.head(5))


    # st.dataframe(data.head(5))
    chains_numb = st.number_input(label='chians for view', min_value=1, max_value=max_chains, step=1)
    st.plotly_chart(Painter.plot_chains(data_coord, chains_numb))
