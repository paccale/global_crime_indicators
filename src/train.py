import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor



from data_preprocessing import (
    load_data, align_wdi_countries_names, select_wdi_indicators, 
    add_continents_to_wdi, handling_wdi_missing_values, rename_columns, 
    merge_datasets, get_scaler, scale_features,
    get_vectorizer, vectorize_df
)
from config import (
    RF_PARAMS, DATA_DIR, MODEL_DIR,
    WDI_2021_PATH, OC_2021_PATH,
    WDI_2023_PATH, OC_2023_PATH,
    RANDOM_STATE
                    )

SAVE_MODEL = True

def prepare_data():
    
    ### TRAINING DATA (2021 Dataset)
    df_wdi, df_oc = load_data(wdi_path=WDI_2021_PATH, oc_path=OC_2021_PATH, year_used=2021)
    oc_countries_set = set(df_oc['Country'].tolist())
    print(oc_countries_set)
    df_wdi = align_wdi_countries_names(df_wdi,oc_countries_set=oc_countries_set)
    df_wdi = select_wdi_indicators(df_wdi_correct_countries=df_wdi, year_used=2021)
    df_wdi = add_continents_to_wdi(df_wdi_filtered=df_wdi, )
    df_wdi = handling_wdi_missing_values(df_wdi_with_continents=df_wdi, )
    df_wdi = rename_columns(df_wdi, suffix='_wdi' ) 
    df_oc = rename_columns(df_oc,suffix='_oc', keep_4_words=False, )
    df_merged = merge_datasets(df_wdi_clean=df_wdi, df_oc=df_oc, )

    X_full_train = df_merged.drop(columns=['criminality_oc', 'country'])
    y_full_train = df_merged['criminality_oc']

    dv = get_vectorizer(X_train=X_full_train)
    X_full_train = vectorize_df(X_full_train, dv=dv)
    scaler = get_scaler(X_full_train)
    X_full_train = scale_features(X=X_full_train, scaler=scaler)
    
    ### TEST DATA (2023 DATASET)
    df_wdi, df_oc = load_data(wdi_path=WDI_2023_PATH, oc_path=OC_2023_PATH, year_used=2023)
    df_wdi = align_wdi_countries_names(df_wdi,oc_countries_set=oc_countries_set)
    print(f'PASQUALE: {df_wdi.shape}')
    df_wdi = select_wdi_indicators(df_wdi_correct_countries=df_wdi, year_used=2023)
    df_wdi = add_continents_to_wdi(df_wdi_filtered=df_wdi, )
    df_wdi = handling_wdi_missing_values(df_wdi_with_continents=df_wdi, )
    df_wdi = rename_columns(df_wdi, suffix='_wdi' ) 
    df_oc = rename_columns(df_oc,suffix='_oc', keep_4_words=False, )
    df_merged = merge_datasets(df_wdi_clean=df_wdi, df_oc=df_oc, )
    
    X_test = df_merged.drop(columns=['criminality_oc', 'country'])
    y_test = df_merged['criminality_oc']
    X_test = vectorize_df(X_test, dv=dv)
    X_test = scale_features(X=X_test, scaler=scaler)

        
    return X_full_train, y_full_train, X_test, y_test, dv, scaler
                                       
    
    
    
def train_model(params = RF_PARAMS):
    X_train, y_train, X_test, y_test, dv, scaler = prepare_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    rf_model = RandomForestRegressor(
        **params, 
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    rf_model.fit(X_train,  y_train)
    print(f"Model trained successfully!")
    
    if SAVE_MODEL:
        os.makedirs(MODEL_DIR, exist_ok=True)
        filename = f'{MODEL_DIR}/RandomForest.pkl'
        pickle.dump(rf_model, open(filename, 'wb'))
        print(f"Model Saved in {filename}")
        filename = f'{MODEL_DIR}/dict_vectorizer.pkl'
        pickle.dump(dv, open(filename, 'wb'))
        filename = f'{MODEL_DIR}/scaler.pkl'
        pickle.dump(scaler, open(filename, 'wb'))
    
        

if __name__ == "__main__":
    train_model()