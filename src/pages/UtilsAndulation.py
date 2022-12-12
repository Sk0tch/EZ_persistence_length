import os
import re
import numpy as np
import math as mt
import pandas as pd
import streamlit as st
from random import random
from scipy.stats import linregress
from statsmodels.nonparametric.kernel_regression import KernelReg

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import plotly.io as pio
# pio.templates.default = "ggplot2"
# pio.templates.default = "plotly_white"

from src.pages.Utils import LinearMath

class Chain:
    def __init__(self, x, y, name=None):
        """
        x, y - выборки, np array
        """
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        if name != None:
            self.name = name
        else:
            self.name = 'NoName'

    def calc_rad_cumsum(self, deviation=3):
        """deviation - допустимое отклонение в сигмах"""
        rad     = np.array([0])
        length  = np.array([0])
        vectors = np.array([[1, 0]])
        x = self.x
        y = self.y
        for n in range(1, len(x)):
            v       = (x[n]-x[n-1], y[n]-y[n-1])
            vectors = np.append(vectors, [v], axis=0)
            rad     = np.append(rad, LinearMath.calc_angle(vectors[n-1], vectors[n]))
            length  = np.append(length, LinearMath.get_len(v))
        std = rad.std()
        if deviation != None:
            remove_emissions = np.vectorize(lambda x: x-np.sign(x)*mt.pi*2 if abs(x)>3*std else x)
            rad              = remove_emissions(rad)
        self.length = length.cumsum()
        self.contour_length = self.length.max()
        self.rad_diff = rad
        self.rad    = rad.cumsum()
        self.std    = std

    def kernel_reg(self, step=1, window=None):
        """
        step - шаг интерполяции
        window - окно регрессии
        """
        self.step = step
        x_reg = self.length
        y_reg = self.rad
        if window == None:
            window='cv_ls'
        elif type(window)==int:
            window = [window]
        model = KernelReg(endog=y_reg,
                  exog=x_reg,
                  reg_type='ll',
                  var_type='c',
                  bw=window
                 )
        x_new = [i for i in np.arange(0, int(max(x_reg)), step)]
        mean, mfs = model.fit(x_new)
        self.lenght_opt = x_new
        self.rad_opt = mean

    def find_inflection_points(self):
        d2r = (np.diff(self.rad_opt))
        sign = np.sign(d2r)
        sign = np.insert(sign, 0, [sign[0]]*1, axis=0)
        ind_arr = []
        for n in range(0, len(sign)-1):
            if sign[n]*sign[n+1] == -1:
                ind_arr.append(n)
        ind_arr = np.array(ind_arr)
        self.sign = sign
        self.distance_inflection = ind_arr*self.step

    def inflection_frame(self, min_rad_diff=0, min_dist_diff=0):
        inflection_points = pd.DataFrame(self.distance_inflection, columns=['distance'])
        finish_data = pd.DataFrame()
        finish_data['distance'] = self.lenght_opt
        finish_data['rad'] = self.rad_opt
        finish_data = finish_data.merge(inflection_points, on='distance', how='inner').fillna(0)
        i=1
        ind_list = []
        while i<len(finish_data)-1:
            # if i >= len(finish_data)-1:
            #     break
            dist_diff = abs(finish_data['distance'].iloc[i]-finish_data['distance'].iloc[i-1])
            rad_diff = abs(finish_data['rad'].iloc[i]-finish_data['rad'].iloc[i-1])
            if (dist_diff<min_dist_diff)|(rad_diff<min_rad_diff):
                finish_data = finish_data.drop([i-1, i]).reset_index(drop=True)
                i = 1
                continue
            i += 1
        finish_data['distance_right'] = finish_data['distance'].shift(-1)
        finish_data                   = finish_data.rename(columns={'distance':'distance_left'})
        finish_data['rad_right']      = finish_data['rad'].shift(-1)
        finish_data                   = finish_data.rename(columns={'rad':'rad_left'})
        finish_data['distance_diff']  = (finish_data['distance_right'] - finish_data['distance_left']).abs()
        finish_data['rad_diff']       = (finish_data['rad_right'] - finish_data['rad_left']).abs()
        return finish_data.dropna()

    def andulation_interval_frame(self, min_rad_diff=0, max_rad_diff=4, min_dist_diff=0, max_dist_diff=1000):
        # точки перегиба
        df_dist = self.inflection_frame(min_rad_diff=min_rad_diff,
                                      min_dist_diff=min_dist_diff)
        # аппроксимация цепи
        df_coord = self.calc_reg_chain()
        # сначала убираем слишком близкие точки перегиба, потом отсекаем по верхней границе слишком большие андуляции
        df_dist = pd.merge(df_dist, df_coord, how='cross')
        df_dist = df_dist[(df_dist['distance'] < df_dist['distance_right'])
                          & (df_dist['distance'] >= df_dist['distance_left'])]
        df_dist.reset_index(drop=True, inplace=True)
        mask = (
               (df_dist['distance_diff'] <= max_dist_diff)&
               (df_dist['rad_diff'] <= max_rad_diff)
               )
        # тут помечаем отрезки цепи, на которых наблюдается андуляция
        df_dist['andulation'] = mask * 1
        return df_dist

    def calc_reg_chain(self):
        points = [[self.x[0], self.y[0]]]
        step = self.step
        first_vector = [1, 0]
        first_vector_len = LinearMath.get_len(first_vector)
        first_vector[0] = first_vector[0] / first_vector_len * step
        first_vector[1] = first_vector[1] / first_vector_len * step
        vectors = [first_vector]
        for angle in np.diff(self.rad_opt[:]):
            new_vec = LinearMath.turn_vec(vectors[-1], angle)
            vectors.append(new_vec)
            new_point = [points[-1][0] + new_vec[0], points[-1][1] + new_vec[1]]
            points.append(new_point)
        df_coord = pd.DataFrame(points, columns=['x', 'y'])
        df_coord['distance'] = self.lenght_opt
        return df_coord

    def run(self, step=1, window=1, min_rad_diff=0, max_rad_diff=4, min_dist_diff=0, max_dist_diff=1000):
        self.calc_rad_cumsum()
        self.kernel_reg(step=step, window=window)
        self.find_inflection_points()
        return self.andulation_interval_frame(min_rad_diff=min_rad_diff, max_rad_diff=max_rad_diff, min_dist_diff=min_dist_diff, max_dist_diff=max_dist_diff)

# @st.cache
class Chains:
    def __init__(self, chains_coord_df):
        """
        x, y - выборки, np array
        """
        self.chains_coord_df = chains_coord_df
        self.chains_ind = chains_coord_df['chain_ind'].unique()
        chains_list = []
        for ind in self.chains_ind:
            one_chain = self.chains_coord_df[self.chains_coord_df['chain_ind'] == ind].copy()
            chains_list.append(Chain(one_chain['x'],one_chain['y']))
        self.chains_list_obj = chains_list

    def andulation_interval_frame_all_chains(self,
                                             step=1,
                                             window=1,
                                             min_rad_diff=0,
                                             max_rad_diff=4,
                                             min_dist_diff=0,
                                             max_dist_diff=1000):
        df_dist_all_chains = pd.DataFrame()
        contours_length_all = 0
        for ind, ch in enumerate(self.chains_list_obj):
            df = ch.run(step=step,
                        window=window,
                        min_rad_diff=min_rad_diff,
                        max_rad_diff=max_rad_diff,
                        min_dist_diff=min_dist_diff,
                        max_dist_diff=max_dist_diff)
            df['chain_ind'] = ind
            contours_length_all += ch.contour_length
            df_dist_all_chains = pd.concat([df_dist_all_chains, df], ignore_index=True)
        return df_dist_all_chains.reset_index(drop=True), contours_length_all

    @staticmethod
    def distribution(df):
        df = df[df['andulation']==1]
        df = df[df['distance_left']==df['distance']]
        df['effective_andulation'] = df['distance_diff'] / df['rad_diff']
        df = df[['distance_diff', 'rad_diff', 'effective_andulation', 'chain_ind']].reset_index(drop=True)
        return df


def plot_angle_deviation(ch):
    x, y = ch.length, ch.rad
    x_new, y_new = ch.lenght_opt, ch.rad_opt
    # x_new15, mean15 = kernel_reg(x, y, step=step, window=15)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name='Данные', mode='markers'))
    fig.add_trace(
        go.Scatter(x=x_new, y=y_new, name='Регрессия cv_ls', mode='lines'))
    # fig.add_trace(go.Scatter(x=x_new15, y=mean15, name='Регрессия 15',  mode='lines'))
    fig.update_layout(
        title='Отклонение угла вдоль контурной длины от начального вектора',
        xaxis_title="Контурная длина (мкм)",
        yaxis_title="Угол отклонения (рад)",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=450, width=900)
    fig.show()


def plot_source_reg(ch):
    df_coord = ch.calc_reg_chain()
    fig = px.scatter(
        df_coord,
        x='x',
        y='y',
        title='Цепь',
        hover_data=['distance'],
        width=1000,
        height=600,
    )
    fig.add_trace(
        go.Scatter(
            x=ch.x,
            y=ch.y,
            marker=dict(color="purple", size=4),
            mode="markers",
        ))
    fig.update_layout(
        #                   legend_orientation="h",
        #                   legend=dict(x=.5, xanchor="center"),
        margin=dict(l=0, r=0, t=30, b=0),
        #                   height=450,
        #                   width=900
    )
    fig.update_yaxes(
        scaleanchor="y",
        scaleratio=1,
    )
    fig.show()


def show_inflection_points(ch,
                           min_rad_diff=0,
                           min_dist_diff=0,
                           max_dist=100,
                           max_rad=4):
    df_dist = ch.inflection_frame(min_rad_diff=min_rad_diff,
                                  min_dist_diff=min_dist_diff)
    df_dist = df_dist.rename(columns={'distance': 'distance_left'})
    df_coord = ch.calc_reg_chain()
    df_dist = pd.merge(df_dist, df_coord, how='cross')
    df_dist = df_dist[(df_dist['distance'] < df_dist['distance_right'])
                      & (df_dist['distance'] >= df_dist['distance_left'])]
    mask = (
           (df_dist['distance_diff'] <= max_dist)&
           (df_dist['rad_diff'] <= max_rad)
           )
    df_dist['andul'] = mask * 1

    dot_frame = pd.merge(df_dist[['distance_left']],
                         df_coord,
                         left_on='distance_left',
                         right_on='distance',
                         how='inner')

    fig = px.scatter(
        df_dist,
        x='x',
        y='y',
        color='andul',
        title='Цепь с точками перегиба',
        hover_data=['distance'],
    )
    fig.add_trace(
        go.Scatter(
            x=dot_frame['x'],
            y=dot_frame['y'],
            marker=dict(color="red", size=5),
            mode="markers",
            name="Точки перегиба",
            hovertemplate =
                    '<i>x</i>: %{x:.1f}'+
                    '<br>y: %{y:.1f}'+
                    '<br><b>distance</b>: <b>%{text}</b>',
            text = dot_frame['distance'],
        ))
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()

def show_plots_with_andulation(df_dist_all_chains, indx=0):
    df_plot = df_dist_all_chains[df_dist_all_chains['chain_ind']==indx]
    dot_frame = df_plot[df_plot['distance']==df_plot['distance_left']]
    df_andul = df_plot[df_plot['andulation']==1]
    df_noandul = df_plot[df_plot['andulation']==0]
    fig = go.Figure(data=go.Scatter(x=df_andul['x'],
                                    y=df_andul['y'],
                                    marker=dict(color="blue", size=5),
                                    mode="markers",
                                    name="Сегмент андуляции",
                                    hovertemplate =
                                            '<i>x</i>: %{x:.1f}'+
                                            '<br>y: %{y:.1f}'+
                                            '<br><b>distance</b>: <b>%{text}</b>',
                                    text = df_andul['distance'],)) # hover text goes here
    fig.add_trace(
        go.Scatter(
            x=df_noandul['x'],
            y=df_noandul['y'],
            marker=dict(color="black", size=5),
            mode="markers",
            name="Сегмент без андуляции",
            hovertemplate =
                    '<i>x</i>: %{x:.1f}'+
                    '<br>y: %{y:.1f}'+
                    '<br><b>distance</b>: <b>%{text}</b>',
            text = df_noandul['distance'],
        ))
    fig.add_trace(
        go.Scatter(
            x=dot_frame['x'],
            y=dot_frame['y'],
            marker=dict(color="red", size=5),
            mode="markers",
            name="Точка перегиба",
            hovertemplate =
                    '<i>x</i>: %{x:.1f}'+
                    '<br>y: %{y:.1f}'+
                    '<br><b>distance</b>: <b>%{text}</b>',
            text = dot_frame['distance'],
        ))
    # fig.update_layout(
    #     title='Цепи с участками андуляции и точками перегиба',
    #     legend_title="Legend",
    #     )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
                      showlegend=False,
                      template='ggplot2',
                      margin=dict(l=0, r=0, b=0, t=0),
                      width=600,
                      height=300,
    )
    return fig

def add_chain(df_dist_all_chains, fig, indx, row, col):
    df_plot = df_dist_all_chains[df_dist_all_chains['chain_ind']==indx]
    dot_frame = df_plot[df_plot['distance']==df_plot['distance_left']]
    df_andul = df_plot[df_plot['andulation']==1]
    df_noandul = df_plot[df_plot['andulation']==0]
    trace0 = go.Scatter(x=df_andul['x'],
                        y=df_andul['y'],
                        marker=dict(color="blue", size=5),
                        mode="markers",
                        name="Сегменты андуляции",
                        hovertemplate =
                                '<i>x</i>: %{x:.1f}'+
                                '<br>y: %{y:.1f}'+
                                '<br><b>distance</b>: <b>%{text}</b>',
                        text = df_andul['distance'],)
    trace1 = go.Scatter(
                        x=df_noandul['x'],
                        y=df_noandul['y'],
                        marker=dict(color="black", size=5),
                        mode="markers",
                        name="Сегменты без андуляции",
                        hovertemplate =
                                '<i>x</i>: %{x:.1f}'+
                                '<br>y: %{y:.1f}'+
                                '<br><b>distance</b>: <b>%{text}</b>',
                        text = df_noandul['distance'],
                    )
    trace2 = go.Scatter(
                        x=dot_frame['x'],
                        y=dot_frame['y'],
                        marker=dict(color="red", size=5),
                        mode="markers",
                        name="Точки перегиба",
                        hovertemplate =
                                '<i>x</i>: %{x:.1f}'+
                                '<br>y: %{y:.1f}'+
                                '<br><b>distance</b>: <b>%{text}</b>',
                        text = dot_frame['distance'],
                        )
    fig.append_trace(trace0, row, col)
    fig.append_trace(trace1, row, col)
    fig.append_trace(trace2, row, col)

def plots_with_andulation(df_dist_all_chains, count=1):
    st.subheader('Цепи с точками перегиба и секциями андуляции')
    indxs = df_dist_all_chains['chain_ind'].unique()
    indxs = indxs[:count]
    for n, indx in enumerate(indxs):
        st.plotly_chart(show_plots_with_andulation(df_dist_all_chains, indx))

def distribution_show(finish_data, norm, contours_length_all, linear_bins=None, rad_bins=None, effective_bins=None):
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['Линейный размер андуляции', 'Угловой размер андуляции', 'Эффективный радиус андуляции'],
                        specs=[[{"type": 'bar'}, {"type": "bar"}],
                               [{"colspan": 2} , None]],
                        vertical_spacing = 0.15)


    hist0 = np.histogram(finish_data['distance_diff'], bins=linear_bins)
    hist1 = np.histogram(finish_data['rad_diff'], bins=rad_bins)
    hist2 = np.histogram(finish_data['effective_andulation'], bins=effective_bins)

    if norm == 'length':
        norm_on0, norm_on1, norm_on2 = contours_length_all/1000, contours_length_all/1000, contours_length_all/1000
    elif norm == 'perc':
        norm_on0 = sum(hist0[0])/100
        norm_on1 = sum(hist1[0])/100
        norm_on2 = sum(hist2[0])/100
    else:
        norm_on0, norm_on1, norm_on2 = 1, 1, 1

    trace0 = go.Bar(x=hist0[1], y = hist0[0]/norm_on0)
    trace1 = go.Bar(x=hist1[1], y = hist1[0]/norm_on1)
    trace2 = go.Bar(x=hist2[1], y = hist2[0]/norm_on2)
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)
    fig.append_trace(trace2, 2, 1)
    fig.update_layout(
                      showlegend=False,
                      template='ggplot2',
                      margin=dict(l=0, r=0, b=0),
                      width=600,
                      height=600,
    )
    return fig
