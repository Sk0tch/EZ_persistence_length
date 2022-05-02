import os
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import linregress

import plotly.express as px
import plotly.graph_objects as go

from src.pages.Utils import Parser, LinearMath, Painter, DataProcessor, LoadData

MODE_PARAM = {
'Средний квадрат угла':'sq_ang_mean',
'Средний косинус угла':'cos_mean',
'Средний угол':'ang_mean'
}


def approximation(data_src, x_name='distance', y_name='angle', min_x=10, max_x=200):
    data = data_src[(data_src['distance']>=min_x) & (data_src['distance']<=max_x)]
    x = data[x_name]
    y = data[y_name]
    lineregress_values = linregress(x, y)

    slope = lineregress_values.slope
    stderr = lineregress_values.stderr
    intercept = lineregress_values.intercept

    return slope, intercept, stderr

def approximation_block(data):
    group=data.copy()
    min_x, max_x = st.session_state.slider
    slope, intercept, stderr = approximation(data_src=group, x_name='distance', y_name='sq_ang_mean', min_x=min_x, max_x=max_x)
    st.plotly_chart(plot_approximation(group, slope, intercept, stderr, x_name='distance', y_name='sq_ang_mean', min_x=min_x, max_x=max_x))
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.write(f'slope = {slope:.2f} ± {stderr:.2f}')
    with col_a2:
        st.write(f'Persistance len: {1/slope:.2f} ± {1/slope**2 * stderr:.2f} nm')

def plot_approximation(data_group, slope, intercept, stderr, x_name='distance', y_name='angle', min_x=0, max_x=200):
    fig = go.Figure()
    data = data_group[(data_group['distance']>=min_x) & (data_group['distance']<=max_x)].copy()
    fig.add_trace(go.Scatter(
        x=data[x_name],
        y=data[y_name],
        name="data"
    ))

    fig.add_trace(go.Scatter(
        x=[min_x, max_x],
        y=[min_x*slope + intercept, max_x*slope + intercept],
        mode="lines",
        line=dict(color='red', width=2,
                                  dash='dash'),
        name="lineregression"
        ))

    fig.update_xaxes(range=[min_x, max_x])
    fig.update_yaxes(range=[data[y_name].min(), data[y_name].max()])

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

    # fig.add_annotation(x=100, y=0.2,
    #             text=f'p = {1/slope:.2f} ± {1/slope**2 * stderr:.2f}',
    #             showarrow=False,
    #             bordercolor="#c7c7c7",
    #             bgcolor="white",
    #             font=dict(
    #             size=20,
    #             )
    #                   )
    return fig


def update_slider():
    st.session_state.slider = st.session_state.numeric1, st.session_state.numeric2

def update_numin():
    st.session_state.numeric1 = st.session_state.slider[0]
    st.session_state.numeric2 = st.session_state.slider[1]


def PersistenceLen():
    st.header('Persistence length calculation')

    group = LoadData('group')

    st.dataframe(group.head(5))

    value_name = st.selectbox('Способ подсчета',
                                ('Средний квадрат угла', 'Средний косинус угла', 'Средний угол'))

    x = group['distance']
    y = group[MODE_PARAM[value_name]]
    color = group['count']
    st.plotly_chart(Painter.plot_line_color(x, y, color))

    st.subheader('Approximation')

    min = int(group['distance'].min())
    max = int(group['distance'].max())

    min_val = min
    max_val = int(group.loc[group['count']>100, 'distance'].max())


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
        approximation_block(group)
