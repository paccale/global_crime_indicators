""" 
Data Preprocessing pipeline for Crime Prediction
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler
import numpy as np

from src.config import (COUNTRY_NAME_MAPPING_WDI_TO_OC, WDI_INDICATORS, 
                        COUNTRY_TO_CONTINENT_MAPPING, MISSING_SUMMARY, 
                        TARGET_COLUMN)

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


def select_wdi_indicators(df_wdi_correct_countries: pd.DataFrame,year_used, indicator_codes=WDI_INDICATORS ) -> pd.DataFrame:
    """ 
    Function to filter wdi dataset indexes, remove useles columns and pivot the df
    """
    df_wdi_filtered = df_wdi_correct_countries[df_wdi_correct_countries['Indicator Code'].isin(indicator_codes)].copy()
    
    found = set(df_wdi_filtered['Indicator Code'].unique())
    missing = set(indicator_codes) - found
    if missing:
        print(f"\n Indicator not found in WDI:")
        for code in missing:
            print(f"    {code}")
    
    # after columns have been selected, remove useless columns
    del df_wdi_filtered['Country Name']
    del df_wdi_filtered['Country Code']
    
    # put country name on first place
    col = "Country_standardized"
    cols = [col] + [c for c in df_wdi_filtered.columns if c != col]
    df_wdi_filtered = df_wdi_filtered[cols]
    
    df_wdi_filtered = df_wdi_filtered.pivot(index = 'Country_standardized', columns = 'Indicator Name', values = f'{str(year_used)}')
    return df_wdi_filtered


def add_continents_to_wdi(df_wdi_filtered: pd.DataFrame, country_to_continent: dict = COUNTRY_TO_CONTINENT_MAPPING ):
    df_wdi_filtered['continent'] = df_wdi_filtered.index.map(country_to_continent) 
    col = 'continent'
    df_wdi_filtered= df_wdi_filtered[[col] + [c for c in df_wdi_filtered.columns if c != col]] 
    df_wdi_with_continents = df_wdi_filtered.copy()
    return

def handling_wdi_missing_values(df_wdi_with_continents: pd.DataFrame, missing_indexes: list= MISSING_SUMMARY):
    if df_wdi_with_continents.isnull().sum() >0:
        for col in missing_indexes:
            # Fill null for each continent
            for continent in df_wdi_with_continents['continent'].unique():
                if pd.isnull(continent):
                    continue
                # select countries in the continent
                continent_mask = df_wdi_with_continents['continent'] == continent
                # count missing values in this continent
                missing_in_continent = df_wdi_with_continents.loc[continent_mask, col].isnull().sum()
                
                if missing_in_continent > 0:
                    # compute continent mead
                    continent_mean = df_wdi_with_continents.loc[continent_mask, col].mean()
                    
                    # If all the countries of a continent are missing, use global mean
                    if pd.isnull(continent_mean):
                        continent_mean = df_wdi_with_continents[col].mean()
                        print(f"    {continent:20s}: {missing_in_continent:2d} missing → Global mean = {continent_mean:.2f}")
                    else:
                        print(f"    {continent:20s}: {missing_in_continent:2d} missing → Continent mean = {continent_mean:.2f}")
                    
                    # fillna in the df
                    df_wdi_with_continents.loc[continent_mask, col] = df_wdi_with_continents.loc[continent_mask, col].fillna(continent_mean)
                    
            # verify no more missing values
            remaining = df_wdi_with_continents[col].isnull().sum()            
            if remaining > 0:
                print(f"  !!{remaining} missing values → Use global mean ")
                global_mean = df_wdi_with_continents[col].mean()
                df_wdi_with_continents[col].fillna(global_mean, inplace=True)        
                
            # final verify
            total_missing_after = df_wdi_with_continents[missing_indexes.index].isnull().sum().sum()
            assert total_missing_after == 0, f"STILL {total_missing_after} MISSING!!"
            print(f"Sustitution completed: 0 missing values")
    else:
        print(f"Sustitution completed: 0 missing values")
        df_wdi_no_null_values = df_wdi_with_continents.copy()
        return df_wdi_no_null_values
            
            
def rename_columns(df: pd.DataFrame, col: str, suffix: str, remove_punctuation: bool=True, no_white_spaces: bool = True, keep_4_words: bool = True, add_suffix: bool = True ) -> list[str]:
    import re
    columns = []
    for col in df.columns:
        # Remove punctuation
        if remove_punctuation:
            col = re.sub(r"[^\w\s]", "", col)
        
        if no_white_spaces: 
            # substitue white spaces with underscores
            col = re.sub(r"\s+", "_", col.strip())
        if keep_4_words:
            # keep only the first 4 words
            parts = col.split("_")[:4]
            col = "_".join(parts)
        if add_suffix:
            # add '_wdi' suffix
            col = f"{col}_{suffix}"
            
        # final clean
        col = re.sub(r"_+", "_", col).strip("_").lower()
        columns.append(col)
    return columns

# df_oc.columns = [modify_name_cols(c, suffix='_oc', keep_4_words=False, ) for c in df_oc.columns]
# df_wdi_clean.columns = [modify_name_cols(c, suffix='_wdi' ) for c in df_wdi_clean.columns]
# del df_oc['index_oc']

def merge_datasets(df_wdi_clean:  pd.DataFrame, df_oc: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the datasets and remove OC columns except target

    Args:
        df_wdi_clean (pd.DataFrame): _description_
        df_oc (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_merged =  df_wdi_clean.merge(df_oc, left_on='country_standardized_wdi', right_on = 'country_oc', how='left', validate='one_to_one')
    
    # remove not necessary columns
    del df_merged['country_oc']
    del df_merged['continent_oc']
    del df_merged['region_oc']
    
    # rename column
    df_merged = df_merged.rename(columns={'country_standardized_wdi': 'country', 'continent_wdi':'continent'})
    
    # drop OC columns except target, we work with WDI data
    oc_cols_to_drop = [col for col in df_merged.columns if col.endswith('_oc') and col != TARGET_COLUMN]
    df_merged = df_merged.drop(columns=oc_cols_to_drop)
    return df_merged


def get_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Fit and return StandardScaler on training data

    Args:
        X_train (pd.DataFrame): Training features

    Returns:
        StandardScaler: Fitted Scaler
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

def scale_features(X: pd.DataFrame, scaler: StandardScaler, ) -> np.ndarray:
    """
    Scale Features using fittest scaler 

    Args:
        X (pd.DataFrame): features to scale
        scaler (StandardScaler): fitted std scaler

    Returns:
        np.ndarray: Scaled features
    """
    return scaler.transform(X)