import pandas as pd
import numpy as np


from data_preprocessing import (
    load_data, align_wdi_countries_names, select_wdi_indicators, 
    add_continents_to_wdi, handling_wdi_missing_values, rename_columns, 
    merge_datasets, get_scaler, scale_features,
    get_vectorizer, vectorize_df
)

from config import (
    RF_PARAMS, DATA_DIR, MODEL_DIR,
    WDI_2021_PATH, OC_2021_PATH
                    )

def prepare_data():
    df_wdi, df_oc = load_data(wdi_path=WDI_2021_PATH, oc_path=OC_2021_PATH, year_used=2021)
    df_wdi = align_wdi_countries_names(df_wdi,oc_countries_set=set(df_oc['Country'].tolist()))
    df_wdi = select_wdi_indicators(df_wdi_correct_countries=df_wdi, year_used=2021)
    df_wdi = add_continents_to_wdi(df_wdi_filtered=df_wdi, )
    df_wdi = handling_wdi_missing_values(df_wdi_with_continents=df_wdi, )
    df_wdi = rename_columns(df_wdi, suffix='_wdi' ) 
    df_oc = rename_columns(df_oc,suffix='_oc', keep_4_words=False, )
    df_merged = merge_datasets(df_wdi_clean=df_wdi, df_oc=df_oc, )

    X_full_train = df_merged.drop(columns=['criminality_oc', 'country'])
    y_full_train = df_merged['criminality_oc']

    dv = get_vectorizer(X_train=X_full_train)
    X_full_train = vectorize_df(X_full_train)
    scaler = get_scaler(X_full_train)
    X_full_train = scale_features(X=X_full_train, scaler=scaler)
                                       
    
    
    
def train_model():
    prepare_data()
    
if __name__ == "__main__":
    train_model()