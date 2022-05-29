import os
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import linregress

import plotly.express as px
import plotly.graph_objects as go

from src.pages.Utils import Parser, LinearMath, Painter, DataProcessor, LoadData, VAR_NAME

MODE_PARAM = {
'Средний квадрат угла':'sq_ang_mean',
'Логарифм косинуса угла':'ln_cos',
'Средний угол':'ang_mean'
}


def per_sq_ang(slope, stderr):
    return 1/slope, 1/slope**2 * stderr

def per_ln_cos(slope, stderr):
    return -1/(2*slope), 1/(2*slope**2) * stderr

PER_CALC_FUNC = {
'sq_ang_mean':per_sq_ang,
'ln_cos':per_ln_cos,
}


def approximation(data_src, x_name='distance', y_name='angle', min_x=10, max_x=200):
    data = data_src[(data_src[x_name]>=min_x) & (data_src[x_name]<=max_x)]
    x = data[x_name]
    y = data[y_name]
    lineregress_values = linregress(x, y)

    slope = lineregress_values.slope
    stderr = lineregress_values.stderr
    intercept = lineregress_values.intercept

    return slope, intercept, stderr

def plot_approximation(data_group, slope, intercept, stderr, x_name='distance', y_name='angle', min_x=0, max_x=200):
    fig = go.Figure()
    data = data_group[(data_group[x_name]>=min_x) & (data_group[x_name]<=max_x)].copy()
    fig.add_trace(go.Scatter(
        x=data[x_name],
        y=data[y_name],
        name="data",
        mode='lines+markers',
    ))

    fig.add_trace(go.Scatter(
        x=[min_x, max_x],
        y=[min_x*slope + intercept, max_x*slope + intercept],
        mode="lines",
        line=dict(color='red', width=2, dash='dash'),
        name="lineregression",

        ))

    fig.update_xaxes(range=[min_x, max_x])
    fig.update_yaxes(range=[data[y_name].min(), data[y_name].max()])

    fig.update_layout(
                    # title="Средний квадрат угла от длины сегмента",
                    xaxis_title="nm",
                    yaxis_title=VAR_NAME[y_name],
                    width=600,
                    height=600,
                    legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
    ))

    return fig

def approximation_block(data, y_name=None):
    group=data.copy()
    min_x, max_x = st.session_state.slider
    slope, intercept, stderr = approximation(data_src=group, x_name='distance', y_name=y_name, min_x=min_x, max_x=max_x)
    st.plotly_chart(plot_approximation(group, slope, intercept, stderr, x_name='distance', y_name=y_name, min_x=min_x, max_x=max_x))
    col_a1, col_a2 = st.columns(2)
    p_len, p_err = PER_CALC_FUNC[y_name](slope, stderr)
    with col_a1:
        st.write(f'slope = {slope:.2f} ± {stderr:.2f}')
    with col_a2:
        st.write(f'Persistance len: {p_len:.2f} ± {p_err:.2f} nm')



def update_slider():
    st.session_state.slider = st.session_state.numeric1, st.session_state.numeric2

def update_numin():
    st.session_state.numeric1 = st.session_state.slider[0]
    st.session_state.numeric2 = st.session_state.slider[1]


def PersistenceLen():
    st.header('Persistence length calculation')
    value_name = st.selectbox('Способ подсчета',
                                MODE_PARAM.keys())
    group = LoadData('group')
    x = group['distance']
    y = group[MODE_PARAM[value_name]]
    color = group['count']
    st.plotly_chart(Painter.plot_line_color(x, y, color, y_name=VAR_NAME[MODE_PARAM[value_name]]))

    st.subheader('Approximation')

    min = int(group['distance'].min())
    max = int(group['distance'].max())

    min_val = min
    max_val = int(group.loc[group['count']>20, 'distance'].max())


    col1, col2 = st.columns(2)

    with col1:
        st.number_input('min value', value = min_val, key = 'numeric1', on_change = update_slider)

    with col2:
        st.number_input('max value', value = max_val, key = 'numeric2', on_change = update_slider)

    st.slider('slider', min_val, max,
                (min_val, max_val),
                key = 'slider', on_change= update_numin)

    st.button('calc', key='calc_button')

    if st.session_state.calc_button:
        approximation_block(group, y_name=MODE_PARAM[value_name])
