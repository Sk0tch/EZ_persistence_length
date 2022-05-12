import os
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit

import plotly.express as px
import plotly.graph_objects as go

from src.pages.Utils import Parser, LinearMath, Painter, DataProcessor, LoadData

MODE_PARAM = {
'Показательная функция':'R_sq_mean',
}

def exp_function(x, a, b, c):
    return a*(x**b) + c


def plot_approximation(x_data, y_data, popt):

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        name="data"
    ))

    fig.add_trace(go.Scatter(
        x=x_data,
        y=[exp_function(x, popt[0], popt[1], popt[2]) for x in x_data],
        mode="lines",
        line=dict(color='red', width=2, dash='dash'),
        name="fit"
        ))

    fig.update_xaxes(range=[x_data.min(), x_data.max()])
    fig.update_yaxes(range=[y_data.min(), y_data.max()])

    fig.update_layout(
                    # title="Средний квадрат угла от длины сегмента",
                    xaxis_title="nm",
                    yaxis_title="θ<sup>2</sup>",
                    width=600,
                    height=600,
                    legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
    ))

    return fig

def approximation_block(data, y_name=None, x_name='distance'):
    min_x, max_x = st.session_state.slider
    group = data[(data[x_name]>=min_x) & (data[x_name]<=max_x)].copy()
    x = group[x_name]
    y = group[y_name]
    popt, pcov = curve_fit(exp_function, x, y)
    perr = np.sqrt(np.diag(pcov))
    st.plotly_chart(plot_approximation(x, y, popt))
    col_a1, col_a2 = st.columns(2)
    # p_len, p_err = PER_CALC_FUNC[y_name](slope, stderr)
    with col_a1:
        st.write(f'a*(x**b) + c')
        st.write(f'a = {popt[0]:.2f} ± {perr[0]:.2f}')
        st.write(f'b = {popt[1]:.2f} ± {perr[1]:.2f}')
        st.write(f'c = {popt[2]:.2f} ± {perr[2]:.2f}')
        # st.write(f'pcov = {perr}')
    # with col_a2:
    #     st.write(f'Persistance len: {p_len:.2f} ± {p_err:.2f} nm')

def update_slider():
    st.session_state.slider = st.session_state.numeric1, st.session_state.numeric2

def update_numin():
    st.session_state.numeric1 = st.session_state.slider[0]
    st.session_state.numeric2 = st.session_state.slider[1]

def DistributionType():

    st.header('Distribution Type')
    group = LoadData('group')
    st.dataframe(group)

    x_name = 'distance'
    y_name = MODE_PARAM['Показательная функция']
    x = group[x_name]
    y = group[y_name]
    color = group['count']
    st.plotly_chart(Painter.plot_line_color(x, y, color))

    st.subheader('Approximation')
    min = int(group[x_name].min())
    max = int(group[x_name].max())
    min_val = min
    max_val = int(group.loc[group['count']>100, x_name].max())

    col1, col2 = st.columns(2)
    with col1:
        st.number_input('min value', value = min_val, key = 'numeric1', on_change = update_slider)
    with col2:
        st.number_input('max value', value = max_val, key = 'numeric2', on_change = update_slider)

    st.slider('slider', min, max,
                (min_val, max_val),
                key = 'slider', on_change=update_numin)

    st.button('calc', key='calc_button')
    if st.session_state.calc_button:
        approximation_block(group, y_name=y_name)
