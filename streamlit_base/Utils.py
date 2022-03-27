import os
import re
import numpy as np
import math as mt
import pandas as pd
import tqdm.notebook as tn
from random import random
from scipy.stats import linregress

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
sns.set_theme()

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
	def create_df_by_list(list_chain):
	    return pd.DataFrame(list_chain, dtype=float).rename(columns={0:'x', 1:'y'})

	@staticmethod	
	def _avg_angle(vector_list, max_n=None, return_cos=False):
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
	    distance = []
	    for n in (range(max_n)):
	        for i in range(max_point-n):
	            angle.append(LinearMath.get_angle(vector_list[n], vector_list[n+i], return_cos=return_cos))
	            if i == 0:
	                distance.append(0)
	            else:
	                distance.append(distance[-1] + LinearMath.get_len(vector_list[i]))
	    return np.array(distance, dtype=float, copy=False), np.array(angle, dtype=float, copy=False)

	@staticmethod
	def multichain_list_to_frame(chain_list, calc='cos'):   
	    if calc == 'cos':
	        return_cos = True
	    elif calc == 'angle':
	        return_cos = False
	    else:
	        print("error name 'calc'")
	        return None
	    df_final = pd.DataFrame()
	    for n, chain_list in tn.tqdm(enumerate(chain_list)):
	        chain = np.array(chain_list, dtype=float)
	        vectors = []
	        for i in (range(len(chain)-1)):
	            vectors.append(LinearMath.make_vector(chain[i], chain[i+1]))
	        distance, angle = Parser._avg_angle(vectors, return_cos=return_cos)
	        df_tmp = pd.DataFrame(data=np.dstack((distance, angle))[0], columns=['distance', 'angle'])
	        df_tmp['chain_num'] = n
	        df_final = df_final.append(df_tmp)
	    return df_final


class Painter:
	@staticmethod
	def plot_chain(chain_list, max_chains=5):
		fig = plt.figure(figsize=(18, 12))
		for i, chain in enumerate(chain_list):
			chain_df = Parser.create_df_by_list(chain)
			plt.plot(chain_df['x'], chain_df['y'])
			if i == max_chains:
			    break
		return fig

	@staticmethod
	def plot_line_color(x, y, color):
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',  name='angle'
								 , marker=dict(color=color/color.sum()
								                 , colorbar=dict(title="count")
								                 , colorscale='Inferno')
								))

		fig.update_layout(legend_orientation="h",
		                  legend=dict(x=.5, xanchor="center"),
		                  margin=dict(l=0, r=0, t=0, b=0))
		fig.update_traces(hoverinfo="all", hovertemplate="Расстояние: %{x}<br>Значение: %{y}")
		return fig