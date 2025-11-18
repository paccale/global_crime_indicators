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
- **Download**: [WDI Database](https://datatopics.worldbank.org/)
- **Content**: 18 carefully selected socio-economic indicators:
  - **Economic**: GDP per capita, GDP growth, unemployment rates, inflation, trade
  - **Education**: Secondary school enrollment
  - **Demographics**: Population, life expectancy, urbanization, working-age population
  - **Governance**: Control of corruption, government effectiveness, political stability, rule of law, regulatory quality, voice and accountability

**Dataset characteristics**:

- Training: 193 countries (2021 data)
- Testing: 193 countries (2023 data)
- Features: 18 numerical + 1 categorical (continent)
- No target leakage: predictions based solely on socio-economic factors

## Exploratory Data Analysis

Key findings from the EDA (see `notebook.ipynb` for detailed analysis):

1. **Missing Values**: Handled through forward-fill and median imputation strategies
2. **Strong Predictors Identified**:
   - Government effectiveness (correlation: -0.72)
   - Rule of law (correlation: -0.71)
   - Control of corruption (correlation: -0.70)
   - Political stability (correlation: -0.58)

3. **Regional Patterns**: Clear differences in crime levels across continents, with governance quality being the strongest differentiator

4. **Feature Engineering**:
   - Added continent information as categorical feature
   - Normalized country names between datasets
   - Selected 18 most relevant WDI indicators

## Model Development

### Baseline Model

- **Linear Regression**: MAE = 0.592, R² = 0.863 (2023 test set)

### Advanced Models Tested

1. **Random Forest** (selected model)
   - Hyperparameter tuning via GridSearchCV
   - Best params: `n_estimators=300, max_depth=15`
   - **Performance**: MAE = 0.502, RMSE = 0.663, R² = 0.894

2. **XGBoost**
   - Comparable performance with extensive regularization
   - Performance: MAE = 0.510, RMSE = 0.672, R² = 0.891

### Model Selection Rationale

**Random Forest was selected** for deployment based on:

- Slightly better MAE (0.502 vs 0.510)
- Better interpretability through feature importance
- Lower computational overhead for inference
- Robust performance across different regions

### Cross-Validation

- 5-fold CV performed on 2021 training data
- Validated on completely separate 2023 test set (temporal validation)

### Feature Importance (Top 10)

1. Government Effectiveness (0.145)
2. Rule of Law (0.118)
3. Control of Corruption (0.113)
4. Political Stability (0.091)
5. Regulatory Quality (0.089)
6. Voice and Accountability (0.061)
7. GDP per capita (0.060)
8. Urban Population % (0.044)
9. Life Expectancy (0.041)
10. School Enrollment (0.036)

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
python train.py
```

This creates model artifacts in `Data/models/`:

- `RandomForest.pkl`
- `dict_vectorizer.pkl`
- `scaler.pkl`

5. **Start the Flask service**

```bash
python app.py
```

Service runs on `http://localhost:5000`

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

**Low criminality country (Belgium-like profile)**:

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
```

Expected criminality score: **~2.5** (low crime)

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
