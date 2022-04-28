import os
import streamlit as st
import numpy as np
import pandas as pd

import plotly.express as px
from src.pages.Utils import Parser, LinearMath, Painter, DataProcessor


DATA_MAP = {
    'angles':'angles.csv',
    'coord':'chains_coord.csv'
}


def LoadData(data_type='angles'):
    return pd.read_csv(os.path.join(st.session_state.data_folder_path, DATA_MAP[data_type]))


def EDA():
    st.header('Exploratory Data Analysis')

    data = LoadData('coord')
    max_chains = data['chain_ind'].max()
    st.write('chains: ', max_chains)
    st.dataframe(data.head(5))
    chains_numb = st.number_input(label='chians for view', min_value=1, max_value=max_chains, step=1)
    st.plotly_chart(Painter.plot_chains(data, chains_numb))
