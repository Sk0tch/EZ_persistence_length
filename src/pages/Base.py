import streamlit as st
import os
import pandas as pd
import datetime as dt

from src.pages.Utils import Parser, DataProcessor


def DataInput():
    uploaded_file = st.file_uploader("Choose .dat file")
    if uploaded_file is not None:
        bytes_data = uploaded_file.read().decode("utf-8")
        chains_coord_list = Parser.parse_data(bytes_data)

        chains_coord_df = Parser.list_chains_to_df(chains_coord_list)
        chains_coord_df.to_csv(os.path.join(st.session_state.data_folder_path, 'chains_coord.csv'), index=None)

        angle_len_df = DataProcessor.angle_len_df(chains_coord_list)
        angle_len_df.to_csv(os.path.join(st.session_state.data_folder_path, 'angles.csv'), index=None)

        return True
    return False

def BaseParamsInput():
    st.header('Data upload')

    is_data_upload = DataInput()

    return is_data_upload
