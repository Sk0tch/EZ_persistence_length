import os
import json
import streamlit as st
import datetime
import hashlib

from src.pages.Base import BaseParamsInput as BaseParamsInput
from src.pages.EDA import EDA as EDA
from src.pages.PersistenceLen import PersistenceLen as PersistenceLen
from src.pages.DistributionType import DistributionType as DistributionType
from src.pages.Andulation import Andulation as Andulation
from src.pages.Utils import Parser, LinearMath, Painter

PAGES = {
    'EDA': EDA,
    'Персистентная длина': PersistenceLen,
    'Тип распределения': DistributionType,
    'Андуляции': Andulation,
}

def InitializeUser():
    if not os.path.exists('data'): os.mkdir('data')
    if 'session_id' not in st.session_state.keys():
        st.session_state.session_id = str(abs(hash(datetime.datetime.now())))
        st.session_state.data_folder_path = os.path.join('data', st.session_state.session_id)
        os.mkdir(st.session_state.data_folder_path)

def Navigation():
    st.sidebar.title('Анализ полимера')
    selection = st.sidebar.radio('Перейти к', list(PAGES.keys()))
    page = PAGES[selection]

    return page

def Main():
    InitializeUser()
    current_page = Navigation()
    is_data_loaded = BaseParamsInput()
    if is_data_loaded:
        current_page()

if __name__ == "__main__":
    Main()
