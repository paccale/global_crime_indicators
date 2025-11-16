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
            print(f"Scaler loaded from {scaler_path}")
            
            
        except FileNotFoundError as e:
            raise Exception(f"Model files not found. Please train the model first. Error: {e}")
        
        except Exception as e:
            raise Exception(f"Error loading: {e}")
        
    
    def preprocess_input(self, input_data: dict) -> pd.DataFrame:
        # X = pd.DataFrame([input_data])
        
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
    model = CrimePredictionModel()
    return model.predict(input_data)
    
    
if __name__ == "__main__":
    print("Tesing crime model")
    test_input = {
 'continent': 'Americas',
 'school_enrollment_secondary_gross_wdi': 83.62,
 'trade_of_gdp_wdi': 106.53,
 'inflation_consumer_prices_annual_wdi': 4.39,
 'unemployment_total_of_total_wdi': 8.27,
 'unemployment_youth_total_of_wdi': 18.77,
 'gdp_per_capita_current_wdi': 7460.0,
 'gdp_growth_annual_wdi': 1.15,
 'government_effectiveness_estimate_wdi': -0.38,
 'control_of_corruption_estimate_wdi': -0.23,
 'regulatory_quality_estimate_wdi': -0.43,
 'population_total_wdi': 411106.0,
 'population_ages_1564_of_wdi': 68.17,
 'political_stability_and_absence_wdi': 0.59,
 'life_expectancy_at_birth_wdi': 73.57,
 'rule_of_law_estimate_wdi': -0.64,
 'urban_population_of_total_wdi': 46.61,
 'voice_and_accountability_estimate_wdi': 0.54,
    }
    
    try: 
        model = CrimePredictionModel()
        prediction = model.predict(test_input)
        print(f"Predicted Criminality score: {prediction:.2f}")
        print(f"Expedcted value around 4.87")
    except Exception as e:
        print(f"\n Prediction failed: {e}")