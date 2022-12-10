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

from src.pages.Utils import LoadData, convert_df
from src.pages.UtilsAndulation import Chain, Chains, show_plots_with_andulation, distribution_show, plots_with_andulation

NORM_DICT = {
    'Нормировка на контурную длину / 1000':'length',
    'Без нормировки':None,
    'Процент':'perc'
}
# @st.cache
def chains_load():
    coord = LoadData('coord')
    chns = Chains(coord)
    return chns

def Andulation():
    st.header('Анализ андуляций')
    chns = chains_load()
    with st.form("my_form"):
       col1, col2 = st.columns(2)
       with col1:
           step  =int(st.number_input('Шаг интерполяции', value=1, key='step'))
       with col2:
           window= int(st.number_input('Окно ядерной регрессии (сглаживание)', value=2, key='window'))
       (min_dist_diff, max_dist_diff)  = st.slider('Диапазон контурной длины андуляции', 0, 50, (5, 25), key = 'dist')
       (min_rad_diff, max_rad_diff)    = st.slider('Диапазон угла андуляции', 0., 3.15, (0.5, 6.28), key = 'rad')
       chains_count                    = st.slider('Количество цепей на первом графике', 1, len(chns.chains_ind), 1, key = 'chain_count')
       submitted                       = st.form_submit_button("Пуск")
       if submitted:
           st.write("Шаг интерполяции (мкм):", step, "Окно сглаживания (мкм):", window)
           st.write("Линейный размер андуляции (мкм):", min_dist_diff, "-", max_dist_diff)
           st.write("Угловой размер андуляции:", min_rad_diff, "-", max_rad_diff)
    andulation_df_full, contours_length_all = chns.andulation_interval_frame_all_chains(step=step,
                                                                window=window,
                                                                min_rad_diff=min_rad_diff,
                                                                max_rad_diff=max_rad_diff,
                                                                min_dist_diff=min_dist_diff,
                                                                max_dist_diff=max_dist_diff)
    andulation_ditr_df = Chains.distribution(andulation_df_full)
    # st.plotly_chart(show_plots_with_andulation(andulation_df_full, chains=int(chain_count)-1))
    # st.plotly_chart(plots_with_andulation(andulation_df_full, count=int(chains_count)))
    plots_with_andulation(andulation_df_full, count=int(chains_count))

    st.subheader('Распределения')
    col1, col2 = st.columns(2)
    with col1:
        select_bins = st.selectbox('Выбор количества интервалов', ('Автоматически', 'Вручную'))
    with col2:
        select_norm = st.selectbox('Нормировка', (NORM_DICT.keys()))
    if select_bins == 'Вручную':
        st.write('Количество интервалов:')
        col1, col2, col3 = st.columns(3)
        with col1:
            linear_bins=int(st.number_input('Линейный размер', value=20))
        with col2:
            rad_bins=int(st.number_input('Угловой размер', value=20))
        with col3:
            effective_bins=int(st.number_input('Эффективный радиус', value=20))
    else:
        linear_bins, rad_bins, effective_bins = 'auto', 'auto', 'auto'

    st.plotly_chart(distribution_show(andulation_ditr_df, norm=NORM_DICT[select_norm], contours_length_all=contours_length_all, linear_bins=linear_bins, rad_bins=rad_bins, effective_bins=effective_bins))

    st.subheader('Данные')
    st.dataframe(andulation_ditr_df)
    csv_andul = convert_df(andulation_ditr_df)
    st.download_button(
         label="Загрузить CSV файл",
         data=csv_andul,
         file_name='data_undulation.csv',
         mime='text/csv',
     )
