import os
import json
import streamlit as st
import datetime
import hashlib

from src.pages.Base import BaseParamsInput as BaseParamsInput
from src.pages.EDA import EDA as EDA
from src.pages.PersistenceLen import PersistenceLen as PersistenceLen
# from src.pages.StatCriteria import StatCriteria as StatCriteria
# from src.pages.Bootstrap import Bootstrap as Bootstrap
# from src.pages.SequentialTesting import SequentialTesting as SequentialTesting
import src.pages.Utils as Utils
from src.pages.Utils import Parser, LinearMath, Painter

PAGES = {
    'eda': EDA,
    'persistent lenght': PersistenceLen,
    # 'bootstrap': Bootstrap,
    # 'sequential testing': SequentialTesting
}

def InitializeUser():
    if 'session_id' not in st.session_state.keys():

        st.session_state.load_params = {
            'project': None,
            'start_date': None,
            'end_date': None,
            'platform': None,
            'min_group_size': None
        }

        st.session_state.is_group_loaded = False
        st.session_state.is_data_updated = False
        st.session_state.is_data_calc = False
        st.session_state.session_id = str(abs(hash(datetime.datetime.now())))
        st.session_state.data_folder_path = os.path.join('data', st.session_state.session_id)
        os.mkdir(st.session_state.data_folder_path)

def Navigation():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio('Go to', list(PAGES.keys()))
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
