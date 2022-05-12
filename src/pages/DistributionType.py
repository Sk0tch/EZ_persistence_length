import os
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit

import plotly.express as px
import plotly.graph_objects as go

from src.pages.Utils import Parser, LinearMath, Painter, DataProcessor, LoadData, VAR_NAME

MODE_PARAM = {
'Показательная функция':'R_sq_mean',
}



def exp_function(x, a, b, c):
    return a*(x**b) + c


def plot_approximation(x_data, y_data, popt, x_name='distance', y_name='angle', title=''):
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
                xaxis_title=x_name,
                yaxis_title=VAR_NAME[y_name],
                width=600,
                height=600,
                title=title,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
    )
    return fig

def approximation_block(group, y_name=None, x_name='distance'):
    min_x, max_x = st.session_state.slider
    data = group[(group[x_name]>=min_x) & (group[x_name]<=max_x)].copy()
    x = data[x_name]
    y = data[y_name]
    popt, pcov = curve_fit(exp_function, x, y, p0=(1, 1/2, 1))
    perr = np.sqrt(np.diag(pcov))
    st.plotly_chart(plot_approximation(x, y, popt, x_name=x_name, y_name=y_name, title='approximation R^2(len) by f(x)=a*(x**b) + c'))
    st.write(f'a*(x**b) + c')
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.write(f'a = {popt[0]:.2f} ± {perr[0]:.2f}')
    with col_a2:
        st.write(f'b = {popt[1]:.2f} ± {perr[1]:.2f}')
    with col_a3:
        st.write(f'c = {popt[2]:.2f} ± {perr[2]:.2f}')

def update_slider():
    st.session_state.slider = st.session_state.numeric1, st.session_state.numeric2

def update_numin():
    st.session_state.numeric1 = st.session_state.slider[0]
    st.session_state.numeric2 = st.session_state.slider[1]

def DistributionType():

    st.header('Distribution Type')
    group = LoadData('group')
    angles = LoadData('angles')

    x_name = 'distance'
    y_name = MODE_PARAM['Показательная функция']
    x = group[x_name]
    y = group[y_name]
    color = group['count']
    st.plotly_chart(Painter.plot_line_color(x, y, color, y_name=VAR_NAME[y_name]))

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

    st.button('calc', key='calc_button1')
    if st.session_state.calc_button1:
        approximation_block(group, y_name=y_name)

    st.subheader('N(θ(l)) distribution')
    st.number_input('len for hist (nm)', value=50, key='hist_mun')
    len = st.session_state.hist_mun
    angles['distance'] = angles['distance'].round(-1)
    st.plotly_chart(px.histogram(angles[angles['distance']==len], x='angle', title='N(θ)'))
