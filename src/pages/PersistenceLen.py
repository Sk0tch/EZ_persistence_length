import os
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import linregress

import plotly.express as px
import plotly.graph_objects as go

from src.pages.Utils import Parser, LinearMath, Painter, DataProcessor, LoadData



def calc_avg_angle(data):
    df_work = data.copy()
    df_work['distance'] = df_work['distance'].round(-1).copy()
    group = df_work.groupby('distance').agg({'angle':['mean', 'count'], 'sq_angle':'mean'}).reset_index()
    group.columns = ['distance', 'ang_mean', 'count', 'sq_ang_mean']
    return group


def approximation(data_src, x_name='distance', y_name='angle', min_x=10, max_x=200):
    data = data_src[(data_src['distance']>=min_x) & (data_src['distance']<=max_x)]
    x = data[x_name]
    y = data[y_name]
    lineregress_values = linregress(x, y)

    slope = lineregress_values.slope
    stderr = lineregress_values.stderr
    intercept = lineregress_values.intercept

    return slope, intercept, stderr

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
