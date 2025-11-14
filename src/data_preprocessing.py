""" 
Data Preprocessing pipeline for Crime Prediction
"""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from src.config import COUNTRY_NAME_MAPPING_WDI_TO_OC, WDI_INDICATORS

def load_data(wdi_path: str, oc_path: str, year_used: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load WDI and OC datasets 

    Args:
        wdi_path (str): Path to WDI CSV File
        oc_path (str): Path to OC CSV file

    Returns:
        pd.DataFrame: Merged Dataframe
    """
    
    df_wdi = pd.read_csv(wdi_path)
    df_oc = pd.read_csv(oc_path)
    
    return df_wdi, df_oc


def align_wdi_countries_names(df_wdi: pd.DataFrame, oc_countries_set: set) -> pd.DataFrame:  
    """
    Aligns countries names in wdi Dataset with OC countries

    Args:
        df_wdi (pd.DataFrame): wdi dataframe from csv file (2021 or 2023)
        oc_countries_set (set): set generated from df_oc['Country] column

    Returns:
        pd.DataFrame: df_wdi with correct countries' names
    """
    df_wdi['Country_standardized'] = df_wdi['Country Name'].replace(COUNTRY_NAME_MAPPING_WDI_TO_OC)
    df_wdi_correct_countries = df_wdi[df_wdi['Country_standardized'].isin(oc_countries_set)].copy()
    return df_wdi_correct_countries

