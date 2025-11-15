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

# These indicators have been selected after the EDA studio on notebook
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

COUNTRY_TO_CONTINENT_MAPPING = {'Afghanistan': 'Asia',
    'Libya': 'Africa',
    'Myanmar': 'Asia',
    'Yemen': 'Asia',
    'Central African Republic': 'Africa',
    'Somalia': 'Africa',
    'Korea, DPR': 'Asia',
    'Venezuela': 'Americas',
    'South Sudan': 'Africa',
    'Syria': 'Asia',
    'Nicaragua': 'Americas',
    'Burundi': 'Africa',
    'Turkmenistan': 'Asia',
    'Equatorial Guinea': 'Africa',
    'Eritrea': 'Africa',
    'Congo, Dem, Rep,': 'Africa',
    'Mali': 'Africa',
    'Comoros': 'Africa',
    'Chad': 'Africa',
    'Haiti': 'Americas',
    'Tajikistan': 'Asia',
    'Guinea-Bissau': 'Africa',
    'Sudan': 'Africa',
    'Suriname': 'Americas',
    'Iran': 'Asia',
    'Zimbabwe': 'Africa',
    'Guinea': 'Africa',
    'Mauritania': 'Africa',
    'Cameroon': 'Africa',
    'Iraq': 'Asia',
    'El Salvador': 'Americas',
    'Belarus': 'Europe',
    'Liberia': 'Africa',
    'Gabon': 'Africa',
    'Congo, Rep,': 'Africa',
    'Mozambique': 'Africa',
    'Papua New Guinea': 'Oceania',
    'Belize': 'Americas',
    'eSwatini': 'Africa',
    'Madagascar': 'Africa',
    'Turkey': 'Asia',
    'Paraguay': 'Americas',
    'Sri Lanka': 'Asia',
    'Lebanon': 'Asia',
    'Laos': 'Asia',
    'Burkina Faso': 'Africa',
    'Niger': 'Africa',
    'Benin': 'Africa',
    'Cambodia': 'Asia',
    'Russia': 'Europe',
    'Kyrgyzstan': 'Asia',
    'Egypt': 'Africa',
    'Timor-Leste': 'Asia',
    'Uganda': 'Africa',
    'Bosnia and Herzegovina': 'Europe',
    'Uzbekistan': 'Asia',
    'Moldova': 'Europe',
    'Lesotho': 'Africa',
    'Pakistan': 'Asia',
    'Azerbaijan': 'Asia',
    'Nepal': 'Asia',
    'Saudi Arabia': 'Asia',
    'Guyana': 'Americas',
    'Sierra Leone': 'Africa',
    'Honduras': 'Americas',
    'Guatemala': 'Americas',
    'Maldives': 'Asia',
    'Tanzania': 'Africa',
    'Mexico': 'Americas',
    'Philippines': 'Asia',
    'Indonesia': 'Asia',
    'Kiribati': 'Oceania',
    'Peru': 'Americas',
    'Algeria': 'Africa',
    'Djibouti': 'Africa',
    'Bangladesh': 'Asia',
    'Cyprus': 'Europe',
    'Angola': 'Africa',
    'Togo': 'Africa',
    'Tunisia': 'Africa',
    'Ukraine': 'Europe',
    'Zambia': 'Africa',
    'Kazakhstan': 'Asia',
    'Namibia': 'Africa',
    'Malawi': 'Africa',
    'Antigua and Barbuda': 'Americas',
    'Brunei': 'Asia',
    'Morocco': 'Africa',
    'Panama': 'Americas',
    'Bhutan': 'Asia',
    'Montenegro': 'Europe',
    'Ethiopia': 'Africa',
    'Vietnam': 'Asia',
    'Thailand': 'Asia',
    'Dominican Republic': 'Americas',
    'Bolivia': 'Americas',
    'Ecuador': 'Americas',
    'Hungary': 'Europe',
    'Brazil': 'Americas',
    'Sao Tome and Principe': 'Africa',
    'Serbia': 'Europe',
    'St, Kitts and Nevis': 'Americas',
    'Nauru': 'Oceania',
    'Solomon Islands': 'Oceania',
    'United Arab Emirates': 'Asia',
    "Côte d'Ivoire": 'Africa',
    'Albania': 'Europe',
    'Malta': 'Europe',
    'Grenada': 'Americas',
    'Vanuatu': 'Oceania',
    'Greece': 'Europe',
    'Gambia': 'Africa',
    'Dominica': 'Americas',
    'Oman': 'Asia',
    'Mongolia': 'Asia',
    'Seychelles': 'Africa',
    'San Marino': 'Europe',
    'Georgia': 'Asia',
    'North Macedonia': 'Europe',
    'Slovakia': 'Europe',
    'Tonga': 'Oceania',
    'Kenya': 'Africa',
    'Bulgaria': 'Europe',
    'Trinidad and Tobago': 'Americas',
    'Palau': 'Oceania',
    'Cuba': 'Americas',
    'St, Vincent and the Grenadines': 'Americas',
    'Jamaica': 'Americas',
    'India': 'Asia',
    'Qatar': 'Asia',
    'Bahrain': 'Asia',
    'Fiji': 'Oceania',
    'Ghana': 'Africa',
    'Botswana': 'Africa',
    'Bahamas': 'Americas',
    'Kuwait': 'Asia',
    'Mauritius': 'Africa',
    'Rwanda': 'Africa',
    'Jordan': 'Asia',
    'St, Lucia': 'Americas',
    'Colombia': 'Americas',
    'South Africa': 'Africa',
    'Costa Rica': 'Americas',
    'Monaco': 'Europe',
    'China': 'Asia',
    'Armenia': 'Asia',
    'Nigeria': 'Africa',
    'Senegal': 'Africa',
    'Marshall Islands': 'Oceania',
    'Samoa': 'Oceania',
    'Poland': 'Europe',
    'Micronesia (Federated States of)': 'Oceania',
    'Malaysia': 'Asia',
    'Croatia': 'Europe',
    'Argentina': 'Americas',
    'Romania': 'Europe',
    'Slovenia': 'Europe',
    'Israel': 'Asia',
    'Tuvalu': 'Oceania',
    'Barbados': 'Americas',
    'Chile': 'Americas',
    'Czech Republic': 'Europe',
    'Italy': 'Europe',
    'Portugal': 'Europe',
    'Cabo Verde': 'Africa',
    'Spain': 'Europe',
    'France': 'Europe',
    'Switzerland': 'Europe',
    'Belgium': 'Europe',
    'United States': 'Americas',
    'Canada': 'Americas',
    'Lithuania': 'Europe',
    'Ireland': 'Europe',
    'Japan': 'Asia',
    'Australia': 'Oceania',
    'Netherlands': 'Europe',
    'Sweden': 'Europe',
    'Germany': 'Europe',
    'Austria': 'Europe',
    'Uruguay': 'Americas',
    'Luxembourg': 'Europe',
    'United Kingdom': 'Europe',
    'Latvia': 'Europe',
    'Singapore': 'Asia',
    'Estonia': 'Europe',
    'New Zealand': 'Oceania',
    'Norway': 'Europe',
    'Andorra': 'Europe',
    'Korea, Rep,': 'Asia',
    'Denmark': 'Europe',
    'Iceland': 'Europe',
    'Liechtenstein': 'Europe',
    'Finland': 'Europe'}

MISSING_SUMMARY = ['School enrollment, secondary (% gross)',
 'Trade (% of GDP)',
 'Inflation, consumer prices (annual %)',
 'Unemployment, total (% of total labor force) (modeled ILO estimate)',
 'Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)',
 'GDP per capita (current US$)',
 'GDP growth (annual %)']


# Model Config
RANDOM_STATE = 42
TEST_SIZE =0.2

# Target Column
TARGET_COLUMN = 'criminality_oc'

# Columns to drop
DROP_COLUMNS = ['country', TARGET_COLUMN]
