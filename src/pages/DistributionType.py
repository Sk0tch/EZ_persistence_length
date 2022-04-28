import os
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import linregress

import plotly.express as px
import plotly.graph_objects as go

from src.pages.Utils import Parser, LinearMath, Painter, DataProcessor, LoadData



def PersistenceLen():
    # if st.session_state.is_data_calc != True:
    st.header('Persistence length calculation')

    # if st.session_state.is_group_loaded == False:
    data = LoadData('angles')
    data['sq_angle'] = data['angle']**2
    # st.dataframe(data.head(5))
    group = calc_avg_angle(data)
    st.dataframe(group.head(5))
    # st.dataframe(group.head(5))
    # st.session_state.is_group_loaded = True

    x = group['distance']
    y = group['sq_ang_mean']
    color = group['count']
    st.plotly_chart(Painter.plot_line_color(x, y, color))

    st.subheader('Approximation')

    min = int(group['distance'].min())
    max = int(group['distance'].max())

    min_val = min
    max_val = max
    # user_values_range = min_val, max_val
    # user_values_range = st.slider(
    #             'Select a range of values',
    #             min_val, max_val, (min_val, max_val))
    # user_values_range[0] = st.number_input(label='min value', min_value=min_val, max_value=max_val, value=min_val)
    # user_values_range[1] = st.number_input(label='max value', min_value=min_val, max_value=max_val, value=max_val)
    # st.session_state.is_data_calc = True

    min_val = st.number_input(label='min value', min_value=min, max_value=max, value=min_val)
    max_val = st.number_input(label='max value', min_value=min, max_value=max, value=max_val)



    user_start = st.button('calc')
    if user_start:
        min_x, max_x = min_val, max_val
        slope, intercept, stderr = approximation(data_src=group, x_name='distance', y_name='sq_ang_mean', min_x=min_x, max_x=max_x)
        st.plotly_chart(plot_approximation(group, slope, intercept, stderr, x_name='distance', y_name='sq_ang_mean', min_x=min_x, max_x=max_x))
        st.write(f'P = {1/slope:.2f} ± {1/slope**2 * stderr:.2f}')
