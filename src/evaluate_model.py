"""
Model Evaluation Script
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from predict import CrimePredictionModel
from data_preprocessing import (
    load_data, align_wdi_countries_names, select_wdi_indicators,
    add_continents_to_wdi, handling_wdi_missing_values, rename_columns,
    merge_datasets
)
from config import WDI_2023_PATH, OC_2023_PATH


def evaluate_model():
    """Valuta il modello sul test set"""
    
    print("=" * 60)
    print("MODEL EVALUATION ON TEST SET (2023 data)")
    print("=" * 60)
    
    # Carica il modello
    print("\nLoading model...")
    model = CrimePredictionModel()
    
    # Prepara i dati di test
    print("\nPreparing test data...")
    df_wdi, df_oc = load_data(wdi_path=WDI_2023_PATH, oc_path=OC_2023_PATH, year_used=2023)
    df_wdi = align_wdi_countries_names(df_wdi, oc_countries_set=set(df_oc['Country'].tolist()))
    df_wdi = select_wdi_indicators(df_wdi_correct_countries=df_wdi, year_used=2023)
    df_wdi = add_continents_to_wdi(df_wdi_filtered=df_wdi)
    df_wdi = handling_wdi_missing_values(df_wdi_with_continents=df_wdi)
    df_wdi = rename_columns(df_wdi, suffix='_wdi')
    df_oc = rename_columns(df_oc, suffix='_oc', keep_4_words=False)
    df_merged = merge_datasets(df_wdi_clean=df_wdi, df_oc=df_oc)
    
    # Prepara X e y
    countries = df_merged['country'].values
    X_test = df_merged.drop(columns=['criminality_oc', 'country'])
    y_true = df_merged['criminality_oc'].values
    
    print(f"Test set size: {len(y_true)} countries")
    
    # Fai predizioni
    print("\nMaking predictions...")
    y_pred = []
    for idx, row in X_test.iterrows():
        input_dict = row.to_dict()
        pred = model.predict(input_dict)
        y_pred.append(pred)
    
    y_pred = np.array(y_pred)
    
    # Calcola metriche
    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"Mean Absolute Error (MAE):  {mae:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"R² Score: {r2:.3f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Analisi degli errori
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    
    errors = y_true - y_pred
    print(f"Mean Error: {np.mean(errors):.3f}")
    print(f"Std Error: {np.std(errors):.3f}")
    print(f"Min Error: {np.min(errors):.3f}")
    print(f"Max Error: {np.max(errors):.3f}")
    
    # Top 10 peggiori predizioni
    print("\n" + "=" * 60)
    print("TOP 10 WORST PREDICTIONS")
    print("=" * 60)
    
    abs_errors = np.abs(errors)
    worst_indices = np.argsort(abs_errors)[-10:][::-1]
    
    print(f"{'Country':<30} {'True':<8} {'Pred':<8} {'Error':<8}")
    print("-" * 60)
    for idx in worst_indices:
        country = countries[idx]
        true_val = y_true[idx]
        pred_val = y_pred[idx]
        error = errors[idx]
        print(f"{country:<30} {true_val:<8.2f} {pred_val:<8.2f} {error:<8.2f}")
    
    # Top 10 migliori predizioni
    print("\n" + "=" * 60)
    print("TOP 10 BEST PREDICTIONS")
    print("=" * 60)
    
    best_indices = np.argsort(abs_errors)[:10]
    
    print(f"{'Country':<30} {'True':<8} {'Pred':<8} {'Error':<8}")
    print("-" * 60)
    for idx in best_indices:
        country = countries[idx]
        true_val = y_true[idx]
        pred_val = y_pred[idx]
        error = errors[idx]
        print(f"{country:<30} {true_val:<8.2f} {pred_val:<8.2f} {error:<8.2f}")
    
    # Distribuzione degli errori per continente
    print("\n" + "=" * 60)
    print("ERROR BY CONTINENT")
    print("=" * 60)
    
    continents = df_merged['continent'].values
    unique_continents = np.unique(continents)
    
    print(f"{'Continent':<15} {'Count':<8} {'MAE':<10} {'RMSE':<10}")
    print("-" * 60)
    for continent in unique_continents:
        mask = continents == continent
        cont_mae = mean_absolute_error(y_true[mask], y_pred[mask])
        cont_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        count = np.sum(mask)
        print(f"{continent:<15} {count:<8} {cont_mae:<10.3f} {cont_rmse:<10.3f}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'y_true': y_true,
        'y_pred': y_pred,
        'countries': countries
    }


if __name__ == "__main__":
    results = evaluate_model()