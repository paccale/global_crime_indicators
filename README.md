# Crime Prediction from Socio-Economic Indicators

A machine learning project that predicts organized crime levels using socio-economic indicators from the World Bank and Global Organized Crime Index.

## Problem Statement

Organized crime poses significant threats to social stability, economic development, and public safety worldwide. Understanding the relationship between socio-economic factors and crime levels can help policymakers identify high-risk regions and allocate resources effectively.

**This project aims to predict criminality scores for countries based on economic, demographic, and governance indicators**, enabling early intervention and informed policy decisions.

**Target users**: Government agencies, international organizations (UN, World Bank), and research institutions focused on crime prevention and social policy.

## Dataset Description

The project combines two authoritative data sources:

### 1. Global Organized Crime Index (2021 & 2023)

- **Source**: [Global Initiative Against Transnational Organized Crime](https://ocindex.net/)
- **Download**: [OCIndex Downloads](https://ocindex.net/downloads)
- **Content**: Criminality scores for 193 countries
- **Target variable**: `criminality_oc` (score ranging from 1-10)

### 2. World Development Indicators (WDI)

- **Source**: [World Bank Open Data](https://databank.worldbank.org/)
- **Download**: [WDI Database](https://www.kaggle.com/datasets/theworldbank/world-development-indicators?select=wdi-csv-zip-57-mb-)
- **Content**: 18 carefully selected socio-economic indicators:
  - **Economic**: GDP per capita, GDP growth, unemployment rates, inflation, trade
  - **Education**: Secondary school enrollment
  - **Demographics**: Population, life expectancy, urbanization, working-age population
  - **Governance**: Control of corruption, government effectiveness, political stability, rule of law, regulatory quality, voice and accountability

**Dataset characteristics**:

I used the 2021 Dataset during the training phase (so for training and validation), while, for testing, the 2023 Dataset.

- Training and Validation: 193 countries (2021 data)
- Testing: 193 countries (2023 data)
- Features: 18 numerical + 1 categorical (continent)
- No target leakage: predictions based solely on socio-economic factors

## Exploratory Data Analysis

Key findings from the EDA (see `notebook.ipynb` for detailed analysis):

1. **Missing Values**: managed through this strategy: replacing the missing data for a country with the average data for countries on the same continent as that country. I thought this would be a more realistic replacement.  

2. **Strong Predictors Identified**:
   - Government effectiveness
   - Rule of law
   - Control of corruption
   - Political stability

3. **Feature Engineering**:
   - Added continent information as categorical feature
   - Normalized country names between datasets
   - Selected 18 most relevant WDI indicators

## Model Development

Different models have been trained and compared.

1. **Linear Regression**:
   - **Performance**: *Train*: MAE = 0.592, RMSE=0.726, R² = 0.863  
*Validation*: MAE = 0.0.629, RMSE=0.0.801, R² = 0.0.704
2. **Decision Tree**
   - Hyperparameter tuning via GridSearchCV
   - Best params: `max_depth=5, min_samples_leaf=4, min_samples_split=2`
   - **Performance**: *Train*: MAE = 0.433, RMSE = 0.540, R² = 0.823  
   *Validation*: MAE = 0.839, RMSE = 0.999, R² = 0.540  (**OVERFITTING IN TRAINING SET**)

3. **Random Forest** (*selected model*)
   - Hyperparameter tuning via GridSearchCV
   - Best params: `n_estimators=300, max_depth=15`
   - **Performance**:*Train*: MAE = 0.236, RMSE = 0.289, R² = 0.949
   *Validation*: MAE = 0.542, RMSE = 0.700, R² = 0.774  
   *Test*: MAE = 0.475, RMSE = 0.607, R² = 0.794

4. **XGBoost**
   - Comparable performance with extensive regularization

### Model Selection Rationale

**Random Forest was selected** for deployment based on:

- Slightly better MAE
- Better interpretability through feature importance
- Lower computational overhead for inference
- Robust performance across different regions

## Project Structure

``` bash
crime-prediction/
│
├── Data/
│   ├── wdi_2021.csv              # Training: WDI indicators (2021)
│   ├── wdi_2023.csv              # Testing: WDI indicators (2023)
│   ├── oc_2021.csv               # Training: Crime scores (2021)
│   ├── oc_2023.csv               # Testing: Crime scores (2023)
│   └── models/
│       ├── RandomForest.pkl      # Trained model
│       ├── dict_vectorizer.pkl   # Feature vectorizer
│       └── scaler.pkl            # Feature scaler
│
├── notebook.ipynb                # Complete EDA and modeling
├── train.py                      # Model training script
├── predict.py                    # Prediction module
├── app.py                        # Flask API (alias: serve.py)
├── data_preprocessing.py         # Data pipeline functions
├── config.py                     # Configuration and constants
├── evaluate_model.py             # Model evaluation utilities
├── test_predict.py               # Unit tests for predictions
├── Dockerfile                    # Docker configuration
└── requirements.txt              # Python dependencies
```

``` bash
├── countries_for_testing.txt
├── Data
│   ├── merged2021.csv
│   ├── merged2023.csv
│   ├── oc_2021.csv
│   ├── oc_2023.csv
│   ├── wdi_2021.csv
│   ├── wdi_2023.csv
│   ├── models
│   │   ├── dict_vectorizer.pkl
│   │   ├── RandomForest.pkl
│   │   └── scaler.pkl
├── Dockerfile
├── graph.png
├── img
│   ├── criminality_score_analysis.png
│   ├── feature_importance_rf.png
│   ├── graph.png
│   ├── linear_regression_performance.png
│   ├── linear_regression_vs_baseline.png
│   └── random_forest_vs_linear_regression.png
├── Instructions.md
├── notebook.ipynb
├── pyproject.toml
├── README.md
├── requirements.txt
├── scores_for_testing.txt
├── src
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   ├── predict.py
│   ├── serve.py
│   └── train.py
├── tests
│   └── test_predict.py
└── uv.lock

```

## Installation & Setup

### Prerequisites

- Python 3.12+
- Docker (for containerized deployment)

### Local Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd crime-prediction
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download data files**

Place the following files in the `Data/` directory:

- `wdi_2021.csv`, `wdi_2023.csv` (World Bank indicators)
- `oc_2021.csv`, `oc_2023.csv` (Crime scores)

4. **Train the model** (optional - pre-trained model included)

```bash
python src/train.py
```

This creates model artifacts in `Data/models/`:

- `RandomForest.pkl`
- `dict_vectorizer.pkl`
- `scaler.pkl`

5. **Start the Flask service**

```bash
python src/serve.py
```

Service runs on `http://localhost:5000`

After starting the server, you can run the script in tests: test_predict.py. 
When asked, insert the name of the country you want to make prediction.

```bash
python tests/test_predict.py
============================================================
CRIME PREDICTION WEB SERVICE - TEST SUITE
============================================================

=== Testing /ping endpoint ===
Status Code: 200
Response: {
  "message": "Crime Prediction Service is running",
  "status": "ok"
}

=== Testing /predict endpoint ===
Insert the country for which you want to predict the Crime Index: Greece
Sending request with Afghanistan 2023 data...
Status Code: 200
Response: {
  "criminality_score": 4.89,
  "status": "success"
}

=== Testing /model_info endpoint ===
Status Code: 200
Response: {
  "features": "[22 features]",
  "model_type": "RandomForestRegressor",
  "n_features": 22,
  "status": "success"
}

============================================================
TEST SUMMARY
============================================================
Health Check............................ ✓ PASSED
Prediction.............................. ✓ PASSED
Model Info.............................. ✓ PASSED
============================================================
```

I have created two useful files for testing in the same test folder: countries_for_testing.txt and scores_for_testing.txt.  
The first contains all the features for each state, which you can copy into a curl request. The second contains all the states and their actual crime indices (both files refer to data from 2023).

You can also simply make the classic curl request that you can find in section API Usage of this readme


## Docker Deployment

### Build the Docker image

```bash
docker build -t crime-prediction .
```

### Run the container

```bash
docker run -p 5000:5000 crime-prediction
```

The API will be available at `http://localhost:5000`

## API Usage

### 1. Health Check

```bash
curl http://localhost:5000/ping
```

**Response:**

```json
{
  "status": "ok",
  "message": "Crime Prediction Service is running"
}
```

### 2. Get Model Information

```bash
curl http://localhost:5000/model_info
```

**Response:**

```json
{
  "model_type": "RandomForestRegressor",
  "n_features": 19,
  "status": "success"
}
```

### 3. Make Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "continent": "Europe",
    "school_enrollment_secondary_gross_wdi": 95.5,
    "trade_of_gdp_wdi": 120.3,
    "inflation_consumer_prices_annual_wdi": 2.1,
    "unemployment_total_of_total_wdi": 6.5,
    "unemployment_youth_total_of_wdi": 15.2,
    "gdp_per_capita_current_wdi": 35000.0,
    "gdp_growth_annual_wdi": 2.5,
    "government_effectiveness_estimate_wdi": 1.2,
    "control_of_corruption_estimate_wdi": 1.5,
    "regulatory_quality_estimate_wdi": 1.3,
    "population_total_wdi": 10000000.0,
    "population_ages_1564_of_wdi": 66.0,
    "political_stability_and_absence_wdi": 1.0,
    "life_expectancy_at_birth_wdi": 80.5,
    "rule_of_law_estimate_wdi": 1.4,
    "urban_population_of_total_wdi": 75.0,
    "voice_and_accountability_estimate_wdi": 1.2
  }'
```

**Response:**

```json
{
  "criminality_score": 3.45,
  "status": "success",
  "input_features": ["continent", "school_enrollment_secondary_gross_wdi", ...]
}
```

### Example Test Cases

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "continent": "Europe",
    "school_enrollment_secondary_gross_wdi": 115.0,
    "trade_of_gdp_wdi": 170.5,
    "inflation_consumer_prices_annual_wdi": 2.3,
    "unemployment_total_of_total_wdi": 5.5,
    "unemployment_youth_total_of_wdi": 13.8,
    "gdp_per_capita_current_wdi": 47000.0,
    "gdp_growth_annual_wdi": 1.8,
    "government_effectiveness_estimate_wdi": 1.6,
    "control_of_corruption_estimate_wdi": 1.7,
    "regulatory_quality_estimate_wdi": 1.5,
    "population_total_wdi": 11500000.0,
    "population_ages_1564_of_wdi": 64.5,
    "political_stability_and_absence_wdi": 1.1,
    "life_expectancy_at_birth_wdi": 81.5,
    "rule_of_law_estimate_wdi": 1.5,
    "urban_population_of_total_wdi": 98.0,
    "voice_and_accountability_estimate_wdi": 1.4
  }'

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
  "continent": "Oceania", 
  "school_enrollment_secondary_gross_wdi": 134.099243164062, 
  "trade_of_gdp_wdi": 49.225184059438064, 
  "inflation_consumer_prices_annual_wdi": 5.597014925373134, 
  "unemployment_total_of_total_wdi": 3.668, 
  "unemployment_youth_total_of_wdi": 8.375, 
  "gdp_per_capita_current_wdi": 64835.9199754535, 
  "gdp_growth_annual_wdi": 3.4419922000902545, 
  "government_effectiveness_estimate_wdi": 1.58987963199615, 
  "control_of_corruption_estimate_wdi": 1.78120493888855, 
  "regulatory_quality_estimate_wdi": 1.94220554828644, 
  "population_total_wdi": 26652777.0, 
  "population_ages_1564_of_wdi": 64.5856920862796, 
  "political_stability_and_absence_wdi": 0.91705721616745, 
  "life_expectancy_at_birth_wdi": 83.05121951219513, 
  "rule_of_law_estimate_wdi": 1.52318394184113, 
  "urban_population_of_total_wdi": 86.617, 
  "voice_and_accountability_estimate_wdi": 1.50660192966461}'
```

## Model Performance Summary

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | 0.592 | 0.785 | 0.863 |
| **Random Forest** | **0.502** | **0.663** | **0.894** |
| XGBoost | 0.510 | 0.672 | 0.891 |

### Regional Performance (Test Set 2023)

The model shows excellent performance across all continents:

- **Africa**: MAE = 0.464 (54 countries)
- **Americas**: MAE = 0.584 (34 countries)
- **Asia**: MAE = 0.528 (48 countries)
- **Europe**: MAE = 0.500 (46 countries)
- **Oceania**: MAE = 0.449 (11 countries)

### Best Predictions (MAE < 0.1)

Countries with exceptional prediction accuracy:

- Belgium: 0.005
- Romania: 0.008
- Albania: 0.016
- Pakistan: 0.025
- Bangladesh: 0.030

## Evaluation Metrics

- **MAE (Mean Absolute Error)**: Primary metric - average prediction error in criminality score units
- **RMSE**: Emphasizes larger errors
- **R²**: Proportion of variance explained (89.4% on test set)

The model achieves **MAE of 0.502** on the 2023 test set, meaning predictions are typically within ±0.5 points of the actual criminality score (on a 1-10 scale).

## Technologies Used

- **Python 3.12**: Core language
- **scikit-learn**: Machine learning (RandomForest, preprocessing)
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **Flask**: REST API framework
- **Docker**: Containerization
- **XGBoost**: Gradient boosting (comparison model)

## Reproducibility

### Complete Workflow

```bash
# 1. Train the model
python train.py

# 2. Evaluate performance
python evaluate_model.py

# 3. Test predictions
python test_predict.py

# 4. Start the service
python app.py
```

### Testing

Run unit tests:

```bash
python test_predict.py
```

Tests cover:

- Model loading
- Preprocessing pipeline
- Prediction accuracy
- API endpoints

## Known Limitations & Future Work

### Current Limitations

1. **Temporal Validation Only**: Model trained on 2021, tested on 2023 - no true out-of-sample countries
2. **Missing Data**: Some countries have incomplete WDI indicators (handled via imputation)
3. **Governance Indicators**: Heavy reliance on World Bank governance scores which may have their own biases
4. **Regional Variations**: Model performs slightly worse for Americas (MAE 0.584 vs 0.449-0.528 for other continents)

### Future Improvements

- [ ] Incorporate additional data sources (e.g., conflict data, drug trafficking routes)
- [ ] Deploy confidence intervals with predictions
- [ ] Create interactive dashboard for policy makers
- [ ] Expand to predict specific crime types (not just aggregate scores)


## Project Context

This project was developed as part of the [ML Zoomcamp 2024](https://github.com/DataTalksClub/machine-learning-zoomcamp) midterm/capstone project. The goal was to build an end-to-end machine learning system including:

- Data preprocessing pipeline
- Model training and evaluation
- REST API deployment
- Docker containerization
- Comprehensive documentation

## Contributing

While this is a course project, suggestions and improvements are welcome. Please open an issue or submit a pull request.

## License

This project is for educational purposes. Data sources (World Bank, Global Organized Crime Index) have their own respective licenses.

## Acknowledgments

- **Data Sources**:
  - [World Bank - World Development Indicators](https://databank.worldbank.org/)
  - [Global Initiative Against Transnational Organized Crime](https://ocindex.net/)
- **Course**: [DataTalks.Club ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- **Inspiration**: Understanding the socio-economic drivers of organized crime for better policy interventions

---

**Author**: Pasquale  
**Project Year**: 2024  
**Last Updated**: November 2024
