import os
import re
import numpy as np
import math as mt
import pandas as pd
import streamlit as st
from random import random
from scipy.stats import linregress

import plotly.express as px
import plotly.graph_objects as go

DATA_MAP = {
    'angles':'angles.csv',
    'coord':'chains_coord.csv',
    'group':'group.csv',
}

VAR_NAME = {
    'sq_ang_mean':'<θ<sup>2</sup>>',
    'ln_cos':'ln(<cos>)',
    'ang_mean':'<θ>',
    'R_sq_mean':'<R<sup>2</sup>>',
    'R_mean':'<R>',
}

def LoadData(data_type='angles'):
    return pd.read_csv(os.path.join(st.session_state.data_folder_path, DATA_MAP[data_type]))


class Painter:
    # @staticmethod
    # def plot_approximation(x_data, y_data, popt, x_name='distance', y_name='angle'):
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(
    #                 x=x_data,
    #                 y=y_data,
    #                 name="data"
    #     ))
    #     fig.add_trace(go.Scatter(
    #                 x=x_data,
    #                 y=[exp_function(x, popt[0], popt[1], popt[2]) for x in x_data],
    #                 mode="lines",
    #                 line=dict(color='red', width=2, dash='dash'),
    #                 name="fit"
    #     ))
    #     fig.update_xaxes(range=[x_data.min(), x_data.max()])
    #     fig.update_yaxes(range=[y_data.min(), y_data.max()])
    #     fig.update_layout(
    #                 xaxis_title=x_name,
    #                 yaxis_title=VAR_NAME[y_name],
    #                 width=600,
    #                 height=600,
    #                 legend=dict(
    #                     yanchor="top",
    #                     y=0.99,
    #                     xanchor="left",
    #                     x=0.01
    #                 )
    #     )
    #     return fig

    @staticmethod
    def plot_chains(chains_df, max_chains=5):
        fig = px.line(chains_df[chains_df['chain_ind']<max_chains], x='x', y='y', color='chain_ind'
                        , title='Show chains'
                        )
        return fig
    @staticmethod
    def plot_line_color(x, y, color, y_name='', x_name='Contour length, nm', title=''):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',  name='angle'
        						 , marker=dict(color=color
    						                 , colorbar=dict(title="count")
    						                 , colorscale='Inferno')
                                 , text=color
        						))
        fig.update_layout(legend_orientation="h",
                          legend=dict(x=.5, xanchor="center"),
                          margin=dict(l=0, r=0, t=0, b=0),
                          xaxis_title=x_name,
                          yaxis_title=y_name,
                          )
        fig.update_traces(hoverinfo="all", hovertemplate="Contour length: %{x} nm<br>Value: %{y}<br>Count:%{text} ")
        return fig


class LinearMath:
    @staticmethod
    def make_vector(p1, p2):
        """вектор из точек"""
        return p2[0] - p1[0], p2[1] - p1[1]

    @staticmethod
    def sum_vectors(v1, v2):
        return v1[0] + v2[0], v1[1] + v2[1]

    @staticmethod
    def get_len(v):
        """длина вектора"""
        return mt.sqrt(v[0] ** 2 + v[1] ** 2)

    @staticmethod
    def get_angle(v1, v2, precision=0.001, return_cos=False):
        """угол между двумя векторами"""
        if return_cos == True:
            cos  = ((v1[0] * v2[0] + v1[1] * v2[1]) /
                (LinearMath.get_len(v1) * LinearMath.get_len(v2)))
            return cos
        else:
            try:
                cos  = ((v1[0] * v2[0] + v1[1] * v2[1]) /
                    (LinearMath.get_len(v1) * LinearMath.get_len(v2)))
                return mt.acos(cos)

            except Exception as error:
                if abs(cos - 1) < precision:
                    cos = 1
                    return mt.acos(cos)
                elif abs(cos + 1) < precision:
                    cos = -1
                    return mt.acos(cos)
                else:
                    print('cos: ', cos, '\n', error)
                    return 0


class Parser:
    @staticmethod
    def parse_data(string_data):
        # file = open(file_path)
        values = string_data.split("\n")
        data = []
        for chain in values:
            chain_split = re.findall(r"\(.*?\)", chain)
            one_chain = []
            if chain_split != []:
                for point in chain_split:
                    one_chain.append(point[1:-1].split(','))
                data.append(one_chain)
        return data

    @staticmethod
    def create_df_by_list(chain_list):
        return pd.DataFrame(chain_list, dtype=float).rename(columns={0:'x', 1:'y'})

    @staticmethod
    def list_chains_to_df(chains_coord_list):
        chains_coord_data = pd.DataFrame()
        for ind, list_chain in enumerate(chains_coord_list):
            chain = pd.DataFrame(list_chain).rename(columns={0:'x', 1:'y'})
            chain['chain_ind'] = ind
            chains_coord_data = chains_coord_data.append(chain)
        return chains_coord_data


class DataProcessor():

    @staticmethod
    def avg_angle(vector_list,
                # max_n=None,
                return_cos=False):
        """
        вычисление среднего угла между векторами массива vector_list
        возвращает DataFrame:
            # возвращает два np массива: расстояние между точками - угол между точками
        """
        # if max_n == None:
        #     max_n = len(vector_list)
        max_point = len(vector_list)

        df = pd.DataFrame()
        cos = []
        distance = []
        R = []
        for n in (range(max_point)):
            vector_end = [0, 0]
            for i in range(max_point-n):
                vector_end = LinearMath.sum_vectors(vector_end, vector_list[n+i])
                R.append(LinearMath.get_len(vector_end))
                cos.append(LinearMath.get_angle(vector_list[n], vector_list[n+i], return_cos=return_cos))
                if i == 0:
                    distance.append(LinearMath.get_len(vector_list[n+i]))
                else:
                    distance.append(distance[-1] + LinearMath.get_len(vector_list[n+i]))
        distance = np.array(distance, dtype=float, copy=False)
        cos = np.array(cos, dtype=float, copy=False)
        R = np.array(R, dtype=float, copy=False)
        R = R**2
        df_tmp = pd.DataFrame(data=np.dstack((distance, cos, R))[0], columns=['distance', 'cos', 'R_sq'])
        return df_tmp

    @staticmethod
    def len_cos_df(chain_list, calc='cos'):
        if calc == 'cos':
            return_cos = True
        elif calc == 'angle':
            return_cos = False
        else:
            print("error name 'calc'")
            return None
        df_final = pd.DataFrame()
        for n, chain_list in enumerate(chain_list):
            chain = np.array(chain_list, dtype=float)
            vectors = []
            for i in (range(len(chain)-1)):
                vectors.append(LinearMath.make_vector(chain[i], chain[i+1]))
            df_tmp = DataProcessor.avg_angle(vectors, return_cos=return_cos)
            df_tmp['chain_ind'] = n
            df_tmp['angle'] = df_tmp['cos'].apply(lambda x: np.arccos(x))
            df_tmp['sq_angle'] = df_tmp['angle']**2
            df_final = df_final.append(df_tmp)
        return df_final

    @staticmethod
    def group_data(data):
        df_work = data.copy()
        df_work['distance'] = ((df_work['distance']/5).round(0)*5).copy()
        group = df_work.groupby('distance').agg({'angle':['count', 'mean', 'std'], 'sq_angle':['mean', 'std'], 'cos':['mean', 'std'], 'R_sq':['mean', 'std']}).dropna().reset_index()
        group.columns = ['distance', 'count', 'ang_mean', 'ang_std', 'sq_ang_mean', 'sq_ang_std', 'cos_mean', 'cos_std', 'R_sq_mean', 'R_sq_std']
        group['ln_cos'] = np.log(group['cos_mean'])
        return group
