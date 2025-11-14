"""
Configuration file for Crime Prediction project
"""
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
MODEL_DIR = DATA_DIR / "models"


# Data files
WDI_2021_PATH = DATA_DIR / "wdi_2021.csv"
WDI_2023_PATH = DATA_DIR / "wdi_2023.csv"
OC_2021_PATH = DATA_DIR / "oc_2021.csv"
OC_2023_PATH = DATA_DIR / "oc_2023.csv"

# Model Path
MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"


COUNTRY_NAME_MAPPING_WDI_TO_OC = {
    'Bahamas, The': 'Bahamas',
    'Brunei Darussalam': 'Brunei',
    'Czechia': 'Czech Republic',
    "Cote d'Ivoire": "Côte d'Ivoire",
    'Egypt, Arab Rep.': 'Egypt',
    'eSwatini': 'Eswatini',
    'Gambia, The': 'Gambia',
    'Iran, Islamic Rep.': 'Iran',
    "Korea, Dem. People's Rep.": 'Korea, DPR',
    'Korea, Rep,': 'Korea, Rep.',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Lao PDR': 'Laos',
    'Micronesia, Fed. Sts.': 'Micronesia (Federated States of)',
    'Russian Federation': 'Russia',
    'Slovak Republic': 'Slovakia',
    'Somalia, Fed. Rep.': 'Somalia',
    'St, Kitts and Nevis': 'St. Kitts and Nevis',
    'St, Lucia': 'St. Lucia',
    'St, Vincent and the Grenadines': 'St. Vincent and the Grenadines',
    'Syrian Arab Republic': 'Syria',
    'Turkiye': 'Turkey',
    'Venezuela, RB': 'Venezuela',
    'Viet Nam': 'Vietnam',
    'Yemen, Rep.': 'Yemen',
    'Congo, Dem, Rep.': 'Congo, Dem. Rep,',
    'Congo, Rep,': 'Congo, Rep.',
}

WDI_INDICATORS = [[
        'NY.GDP.PCAP.CD',           # GDP per capita (current US$) X
        'NY.GDP.MKTP.KD.ZG',        # GDP growth (annual %) X
        'SL.UEM.TOTL.ZS',           # Unemployment, total (% of labor force) X
        'SL.UEM.1524.ZS',           # Youth unemployment (ages 15-24, %) X
        'FP.CPI.TOTL.ZG',           # Inflation, consumer prices (annual %) X
        'NE.TRD.GNFS.ZS',           # Trade (% of GDP) X
    ],

    # Low education and school dropout could push to get involved in crime
    [
        'SE.SEC.ENRR',              # School enrollment, secondary (% gross) X
    ],

    # Weak healtcare systems can create illegal market for drug/organs. Or large young population could indicate more potential recruits
    [
        'SP.DYN.LE00.IN',           # Life expectancy at birth (years) X
        'SP.POP.TOTL',              # Population, total X
        'SP.URB.TOTL.IN.ZS',        # Urban population (% of total) X
        'SP.POP.1564.TO.ZS',        # Population ages 15-64 (% of total) - working age X
    ],

    # These could be the most important, indicating high corruption, weak rule of law or politic instability: each one is a determinant factor in the proliferation of organized crime 
    [
        'CC.EST',                   # Control of Corruption (estimate) X
        'GE.EST',                   # Government Effectiveness (estimate) X
        'PV.EST',                   # Political Stability and Absence of Violence (estimate) X
        'RL.EST',                   # Rule of Law (estimate) X
        'RQ.EST',                   # Regulatory Quality (estimate) X
        'VA.EST',                   # Voice and Accountability (estimate) X
    ],
    ]

# Model Config
RANDOM_STATE = 42
TEST_SIZE =0.2

# Target Column
TARGET_COLUMN = 'criminality_avg_oc'

# Columns to drop
DROP_COLUMNS = ['country', TARGET_COLUMN]
