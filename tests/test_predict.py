"""
Test script for the Crime Prediction Web Service
"""
import requests
import json

import pandas as pd
import ast

# Service URL (modify if running on different host/port)
BASE_URL = "http://localhost:5000"


def test_ping():
    """Test the health check endpoint"""
    print("\n=== Testing /ping endpoint ===")
    response = requests.get(f"{BASE_URL}/ping")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_predict():
    """Test single prediction endpoint"""
    print("\n=== Testing /predict endpoint ===")
    records = []
    with open("tests/countries_for_testing.txt") as f:
        for line in f:
            d = ast.literal_eval(line.strip())     # converte stringa → dict
            # flattiamo il dizionario: {"Belize": {...}} → {"country": "Belize", ...}
            country = list(d.keys())[0]
            values = d[country]
            values["country"] = country
            records.append(values)

    df = pd.DataFrame(records)
    
    # test_data = {
    #     'continent': 'Asia',
    #     'school_enrollment_secondary_gross_wdi': 85.96,
    #     'trade_of_gdp_wdi': 67.58,
    #     'inflation_consumer_prices_annual_wdi': -4.64,
    #     'unemployment_total_of_total_wdi': 13.99,
    #     'unemployment_youth_total_of_wdi': 17.29,
    #     'gdp_per_capita_current_wdi': 413.76,
    #     'gdp_growth_annual_wdi': 2.27,
    #     'government_effectiveness_estimate_wdi': -1.99,
    #     'control_of_corruption_estimate_wdi': -1.15,
    #     'regulatory_quality_estimate_wdi': -1.27,
    #     'population_total_wdi': 41454761.0,
    #     'population_ages_1564_of_wdi': 54.40,
    #     'political_stability_and_absence_wdi': -2.48,
    #     'life_expectancy_at_birth_wdi': 66.04,
    #     'rule_of_law_estimate_wdi': -1.65,
    #     'urban_population_of_total_wdi': 26.93,
    #     'voice_and_accountability_estimate_wdi': -1.85
    # }
    country_name = input("Insert the country for which you want to predict the Crime Index: ")
    test_data = df[df['country']==country_name]
    test_data= test_data.to_dict(orient='records')[0]
    
    print("Sending request with Afghanistan 2023 data...")
    response = requests.post(
        f"{BASE_URL}/predict",
        json=test_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("\n=== Testing /model_info endpoint ===")
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status Code: {response.status_code}")
    result = response.json()
    # Don't print all features if there are many
    if 'features' in result and isinstance(result['features'], list):
        result['features'] = f"[{len(result['features'])} features]"
    print(f"Response: {json.dumps(result, indent=2)}")
    return response.status_code == 200


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("CRIME PREDICTION WEB SERVICE - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_ping),
        ("Prediction", test_predict),
        ("Model Info", test_model_info),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "✓ PASSED" if result else "✗ FAILED"))
        except requests.exceptions.ConnectionError:
            results.append((test_name, "✗ FAILED - Cannot connect to service"))
            print(f"\n⚠️  Cannot connect to {BASE_URL}")
            print("Make sure the Flask service is running:")
            print("  python app.py")
            break
        except Exception as e:
            results.append((test_name, f"✗ FAILED - {str(e)}"))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, result in results:
        print(f"{test_name:.<40} {result}")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()