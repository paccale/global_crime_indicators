# global_crime_indicators

The objective of this project is to analyze and make inferences aboute crime data in countries around the world.

## Datasets

### 1.Global Organized Crime Index

**Source**: [Global Initiative Against Transnational Organized Crime](https://ocindex.net/)  
**Download**: [OCIndex Downloads](https://ocindex.net/downloads)

The Global Organized Crime Index is a comprehensive assessment of organized crime and resilience across 193 UN member states. Published by the Global Initiative Against Transnational Organized Crime, this dataset provides expert-evaluated scores measuring both the scope of criminal activities and countries' capacity to respond.

**Key Features**:

- **193 Countries**  
- **39 Indicators**
  - **Criminality Score** (avg): Overall Organized crime level (1-10 scale)
  - **Criminal Markets** (14 indicators): Human trafficking, drug trade, arms trafficking, wildlife/flora crimes, financial crimes, counterfeiting, extortion, cyber crimes
  - **Criminal Actors** (5 indicators):  Mafia-style groups, criminal networks, state-embedded actors, foreign actors, private sector actors
  - **Resilience** (12 Indicators): Political leadership, governance, judicial systems, law enforcement, anti-money laundering, victim support, prevfention measures  

**Data Quality**: Expert assessments from 2-3 regional specialists per country, ensuring high reliability. Scores range from 1 (low) to 10 (high), with higher criminality scores indicating greater organized crime presence.

**Geographic Coverage**: Global, with regional groupings (Africa, Americas, Asia, Europe, Oceania) for comparative analysis.

### 2.World Development Indicators (WDI)

**Source**: [World Bank Open Data](https://data.worldbank.org/)  
**Download**: [WDI Database](https://datatopics.worldbank.org/world-development-indicators/)  
**Alternative**: [Kaggle Mirror](https://www.kaggle.com/datasets/theworldbank/world-development-indicators)

The World Development Indicators database is the primary World Bank collection of development statistics, compiled from officially recognized international sources. It presents the most current and accurate global development data available.

**Key Features**:
    - **1,500+ indicators** across multiple domains
    - **200+ countries and territories**
    - **Time series data** from 1960 to present (updated annually)

**Selected Indicators for This Analysis**:
    - **Economic**: GDP per capita, GDP growth rate, unemployment rate, inflation
    - **Social**: Gini coefficient (income inequality), poverty headcount ratio, life expectancy, infant mortality rate
    - **Education**: Mean years of schooling, government expenditure on education (% of GDP), literacy rates
    - **Governance**: Government effectiveness, regulatory quality, rule of law indicators
    - **Infrastructure**: Urban population (%), access to electricity, internet users (% population)
    - **Health**: Health expenditure per capita, physicians per 1,000 people

**Data Quality**: Sourced from official national statistical offices, UN agencies, OECD, and other reputable international organizations. The World Bank applies rigorous quality control and harmonization procedures to ensure cross-country comparability.

### Data Integration

The two datasets are merged on country identifiers to create a unified analytical dataset with:
    - **193 observations** (one per country)
    - **50-60 features** (39 crime indicators + 20-30 selected development indicators)
    - **Target variables**: Criminality score (regression), crime categories (classification)
    - **Predictors**: Economic, social, governance, and infrastructure indicators

Missing values are handled through imputation or exclusion depending on indicator availability and analytical requirements.
