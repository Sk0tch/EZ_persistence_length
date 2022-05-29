import os
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.stats import normaltest
from scipy.stats import shapiro

import plotly.express as px
import plotly.graph_objects as go

from src.pages.Utils import Parser, LinearMath, Painter, DataProcessor, LoadData, VAR_NAME

MODE_PARAM = {
'Показательная функция':'R_sq_mean',
}

MODE_ERR = {
'Показательная функция':'R_sq_std',
}



def exp_function(x, a, b, c):
    return a*(x**b) + c

def exp_function_P(x, p):
    e = 2.718281828459045
    # return 4*p*x(1 - 2*p/x * (1 - e**(-x/(2*p))))
    return 4*p*x*(1-2*(p/x)*(1 - e**(-x/(2*p))))

def plot_approximation(x_data, y_data, y_error, popt, x_name='distance', y_name='angle', title=''):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                # error_y=dict(
                #         type='data', # value of error bar given in data coordinates
                #         array=y_error,
                #         visible=True),
                name='data',
                mode='markers',
    ))
    fig.add_trace(go.Scatter(
                x=x_data,
                y=[exp_function_P(x, popt[0]) for x in x_data],
                mode="lines",
                line=dict(color='red', width=2, dash='dash'),
                name="fit"
    ))
    fig.update_traces(marker=dict(size=8),
                  selector=dict(mode='markers'))

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

def approximation_block(group, y_name=None, y_err_name=None, x_name='distance'):
    min_x, max_x = st.session_state.slider
    data = group[(group[x_name]>=min_x) & (group[x_name]<=max_x)].copy()
    x = data[x_name]
    y = data[y_name]
    y_err = data[y_err_name]
    # title = 'approximation R^2(len) by f(x)=4*p*x*(1-2*(p/x)*(1-e**(-x/(2*p))))'
    title = ''
    # popt, pcov = curve_fit(exp_function, x, y, p0=(1, 1/2, 1))
    popt, pcov = curve_fit(exp_function_P, x, y)
    perr = np.sqrt(np.diag(pcov))
    st.plotly_chart(plot_approximation(x, y, y_err, popt, x_name=x_name, y_name=y_name, title=title))
    # st.write('4*p*x(1 - 2*p/x * (1 - e**(-x/(2*p))))')
    st.write(f'p = {popt[0]:.2f} ± {perr[0]:.2f}')

    # st.write(f'a*(x**b) + c')
    # col_a1, col_a2, col_a3 = st.columns(3)
    # with col_a1:
    #     st.write(f'a = {popt[0]:.2f} ± {perr[0]:.2f}')
    # with col_a2:
    #     st.write(f'b = {popt[1]:.2f} ± {perr[1]:.2f}')
    # with col_a3:
    #     st.write(f'c = {popt[2]:.2f} ± {perr[2]:.2f}')

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
    y_err_name = MODE_ERR['Показательная функция']
    x = group[x_name]
    y = group[y_name]
    color = group['count']
    st.plotly_chart(Painter.plot_line_color(x, y, color, y_name=VAR_NAME[y_name]))

    st.subheader('Approximation')
    min = int(group.loc[group['count']>0, x_name].min())
    max = int(group[x_name].max())
    min_val = min
    max_val = int(group.loc[group['count']>50, x_name].max())

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
        approximation_block(group, y_name=y_name, y_err_name=y_err_name)

    st.subheader('N(θ(l)) distribution')
    st.number_input('len for hist (nm)', value=50, key='hist_mun')
    len = st.session_state.hist_mun
    angles['distance'] = angles['distance'].round(-1)
    st.plotly_chart(px.histogram(angles[angles['distance']==len], x='angle', title='N(θ)'))
    data_val = list(angles[angles['distance']==len]['angle'])
    # stat, p = normaltest(data_val)
    stat, p = normaltest(data_val)
    st.write(f"p = {p}")
    print(data_val)
