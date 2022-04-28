import os
import re
import numpy as np
import math as mt
import pandas as pd
import streamlit as st
from random import random
from scipy.stats import linregress

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
sns.set_theme()

DATA_MAP = {
    'angles':'angles.csv',
    'coord':'chains_coord.csv'
}

def LoadData(data_type='angles'):
    return pd.read_csv(os.path.join(st.session_state.data_folder_path, DATA_MAP[data_type]))

class Painter:
	@staticmethod
	def plot_chains(chains_df, max_chains=5):
		fig = px.line(chains_df[chains_df['chain_ind']<max_chains], x='x', y='y', color='chain_ind'
                        , title='Отображение входных данных'
                        )
		return fig

	@staticmethod
	def plot_line_color(x, y, color):
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',  name='angle'
								 , marker=dict(color=color
								                 , colorbar=dict(title="count")
								                 , colorscale='Inferno')
                                 # , title="Средний квадрат угла от длины сегмента"
								))

		fig.update_layout(legend_orientation="h",
		                  legend=dict(x=.5, xanchor="center"),
		                  margin=dict(l=0, r=0, t=0, b=0))
		fig.update_traces(hoverinfo="all", hovertemplate="Расстояние: %{x}<br>Значение: %{y}")
		return fig


class LinearMath:
    @staticmethod
    def make_vector(p1, p2):
        """вектор из точек"""
        return p2[0] - p1[0], p2[1] - p1[1]

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
    def avg_angle(vector_list, max_n=None, return_cos=False):
        """
        вычисление среднего угла между векторами массива vector_list
        возвращает DataFrame:
            возвращает два np массива: расстояние между точками - угол между точками
        """
        if max_n == None:
            max_n = len(vector_list)
        max_point = len(vector_list)

        df = pd.DataFrame()
        angle = []
        distance, segment_len = [], []

        for n in (range(max_n)):
            for i in range(max_point-n):
                angle.append(LinearMath.get_angle(vector_list[n], vector_list[n+i], return_cos=return_cos))
                if i == 0:
                    # segment_len.append(0)
                    distance.append(0)
                else:
                    distance.append(distance[-1] + LinearMath.get_len(vector_list[i]))
                    # segment_len.append(LinearMath.get_len(vector_list[n], vector_list[n+i]))
        return np.array(distance, dtype=float, copy=False), np.array(angle, dtype=float, copy=False)

    @staticmethod
    def angle_len_df(chain_list, calc='angle'):
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
            distance, angle = DataProcessor.avg_angle(vectors, return_cos=return_cos)
            df_tmp = pd.DataFrame(data=np.dstack((distance, angle))[0], columns=['distance', 'angle'])
            df_tmp['chain_num'] = n
            df_final = df_final.append(df_tmp)
        return df_final
