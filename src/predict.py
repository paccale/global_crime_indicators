import pickle 
import pandas as pd
from config import MODEL_DIR

class CrimePredictionModel:
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.load_model()
        
    def load_model(self):
        
        try:
            model_path = f'{MODEL_DIR}/RandomForest.pkl'
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {model_path}")
            
            vectorizer_path = f'{MODEL_DIR}/dict_vectorizer.pkl'
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f"Vectorizer loaded from {vectorizer_path}")
            
            scaler_path = f'{MODEL_DIR}/scaler.pkl'
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded from {vectorizer_path}")
            
            
        except FileNotFoundError as e:
            raise Exception(f"Model files not found. Please train the model first. Error: {e}")
        
        except Exception as e:
            raise Exception(f"Error loading: {e}")
        
    
    def preprocess_input(self, input_data: dict) -> pd.DataFrame:
        df = pd.DataFrame([input_data])
        
        X = self.vectorizer.transform([input_data])
        
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, input_data: dict) -> float:
        """make a prediction for a single input

        Args:
            input_data (dict): dictionary with feature names and values

        Returns:
            float: predicted criminality score
        """
        
        X = self.preprocess_input(input_data)
        prediction = self.model.predict(X)[0]
        return float(prediction)
    
def predict_single(input_data: dict) -> float:
    """Convenience function for single prediction

    Args:
        input_data (dict):  dictionary with feature names and values

    Returns:
        float: predicted criminality score
    """
    
    
if __name__ == "__main__":
    print("Tesing crime model")
    test_input = {
    'continent': 'Asia',
    'school_enrollment_secondary_gross_wdi': 85.96,
    'trade_of_gdp_wdi': 67.58,
    'inflation_consumer_prices_annual_wdi': -4.64,
    'unemployment_total_of_total_wdi': 13.99,
    'unemployment_youth_total_of_wdi': 17.29,
    'gdp_per_capita_current_wdi': 413.76,
    'gdp_growth_annual_wdi': 2.27,
    'government_effectiveness_estimate_wdi': -1.99,
    'control_of_corruption_estimate_wdi': -1.15,
    'regulatory_quality_estimate_wdi': -1.27,
    'population_total_wdi': 41454761.0,
    'population_ages_1564_of_wdi': 54.40,
    'political_stability_and_absence_wdi': -2.48,
    'life_expectancy_at_birth_wdi': 66.04,
    'rule_of_law_estimate_wdi': -1.65,
    'urban_population_of_total_wdi': 26.93,
    'voice_and_accountability_estimate_wdi': -1.85
    }
    
    # try: 
    model = CrimePredictionModel()
    prediction = model.predict(test_input)
    print(f"Predicted Criminality score: {predict_single}")
    print(f"Expedcted valuer around 7.1")
    # except Exception as e:
    #     print(f"\n Prediction failed: {e}")