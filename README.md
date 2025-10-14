# Gentrification Prediction in Seoul Using Transformer-Based Deep Learning

## Overview
<img width="615" height="301" alt="image" src="https://github.com/user-attachments/assets/e3bdc2d0-d4f8-42ba-ada6-e69fd60acf51" />



This project analyzes and predicts gentrification patterns in Seoul's commercial districts using multidimensional socioeconomic and spatial data. By integrating various datasets into a MariaDB-based infrastructure, we identified urban transformation trends and built a predictive model leveraging clustering and deep learning techniques.

## Research Objectives
- Detect patterns of urban gentrification based on commercial district data
- Identify regional clusters showing early signs of socioeconomic transformation
- Develop a data-driven predictive framework for future gentrification risk analysis
- Provide actionable insights for urban policymakers to prevent displacement and support sustainable development

## Dataset Description

### Data Sources
- Seoul Open Data Plaza (서울시 열린 데이터 광장)
- Seoul Commercial District Analysis Service
- Ministry of Land, Infrastructure and Transport (국토교통부)
- Regional business registries

### Data Categories
**Commercial & Economic Indicators:**
- Commercial rent and business turnover rates
- Monthly average income and expenditure by category (food, clothing, medical, transportation, leisure, culture, education, entertainment)
- Business operation duration and closure rates

**Demographic Data:**
- Population distribution by age group (10s, 20s, 30s, 40s, 50s, 60+)
- Living population by time period and day of week
- Gender distribution

**Infrastructure & Spatial Features:**
- Attraction facilities (집객시설): general hospitals, regular hospitals, high schools, universities, department stores, supermarkets, theaters, lodging facilities, subway stations, bus stops
- Building usage and zoning information
- Apartment complex data and average prices

### Data Period
- Training data: Q1 2020 - Q4 2022 (quarterly)
- Prediction target: 2023

### Database Infrastructure
All raw data were normalized and stored in **MariaDB** for efficient querying, integration, and scalability.

## Methodology

### 1. Data Integration & Preprocessing
- Combined heterogeneous data formats (CSV, JSON, API responses) into a unified relational schema using SQL
- Handled missing values through statistical imputation
- Applied MinMaxScaler normalization to standardize feature scales
- Created binary indicators for commercial district types (U: tourist zone, R: traditional market, D: developed commercial area, A: alley commercial area)

### 2. Gentrification Diagnosis Framework
Based on the Korean Ministry of Land, Infrastructure and Transport's gentrification diagnosis system, districts were classified into four stages:

- **Stage 0 (Initial)**: No gentrification or post-gentrification decline; inactive commercial area
- **Stage 1 (Caution)**: Early gentrification signs due to low rent, policy initiatives, or consumption trend changes; capital inflow beginning
- **Stage 2 (Warning)**: Rapid gentrification progress with rent increases, area popularity, and frequent business type changes
- **Stage 3 (Risk)**: Final stage with over-commercialization and involuntary displacement; high probability of commercial decline

### 3. Clustering Analysis
**Method**: K-Means clustering with Elbow Method optimization

**Process:**
- Identified known gentrified areas (Garosu-gil, Hongdae, Sinchon, Ewha, Seongsu, Jongno, Seodaemun) as reference points
- Applied Elbow Method to determine optimal cluster number (K=4)
- Validated clustering quality with Silhouette Score (0.3 for K=4) and PCA visualization
- Cluster distribution: 2,098 / 1,120 / 913 / 732 samples across 4 clusters

**Cluster Characteristics:**
- Cluster 0 (Warning Stage): Highest in attraction facilities, living population, apartment units, and total expenditure
- Cluster 1 (Initial Stage): Low infrastructure but longest average business operation duration
- Cluster 2 (Risk Stage): Low population and facilities but high apartment prices and expenditure; highest gentrification indicator
- Cluster 3 (Caution Stage): High facilities and population but low apartment units; insufficient infrastructure despite capital inflow

### 4. Machine Learning Models

#### Logistic Regression
- Accuracy: 0.592 (59.2%)
- ROC-AUC: Below 0.9 for most classes (excluding class 0)
- Conclusion: Insufficient for multi-class classification with complex features

#### Model Selection with PyCaret 3.2.0
Evaluated multiple regression models using RMSE as the primary metric:

| Model | RMSE |
|-------|------|
| Extra Trees Regressor | **0.0364** |
| Random Forest Regressor | **0.0372** |
| Light Gradient Boosting | 0.0504 |
| Decision Tree Regressor | 0.0456 |

**Selected Models**: Extra Trees Regressor (ET) and Random Forest Regressor (RF)

**Feature Importance Analysis:**

*Extra Trees Regressor Top Features:*
1. Traditional Market (상권 구분 코드 R): 0.2777
2. Developed Commercial Area (상권 구분 코드 D): 0.1717
3. Education Expenditure: 0.0543
4. Medical Expenditure: 0.0505
5. Total Expenditure: 0.0485

*Random Forest Regressor Top Features:*
1. Traditional Market (상권 구분 코드 R): 0.3058
2. Developed Commercial Area (상권 구분 코드 D): 0.1854
3. Education Expenditure: 0.1745
4. Male Living Population: 0.0889
5. Clothing Expenditure: 0.0558

**Key Finding**: Traditional markets and developed commercial areas emerged as the most critical factors, followed by education-related spending patterns.

### 5. Deep Learning: Transformer Model

#### Model Architecture
- Transformer layers: 2
- Attention heads: 4
- Batch size: 64
- Epochs: 1000 (with early stopping after 50 epochs without improvement)
- Optimizer: Adam
- Loss function: Categorical Cross-Entropy with Softmax activation

#### Training Configuration
- Train-test split: 80%-20%
- Data normalization: Applied to all features
- Target encoding: One-hot encoding for 4-class classification
- Input dimensions: 52 features
- Output dimensions: 4 classes (gentrification stages)

#### Performance Metrics
- **Validation Loss**: 0.0092
- **Accuracy**: 99%
- **RMSE**: 0.0641

#### Attention Weight Analysis
The attention mechanism revealed which features the model focused on most:

**Highest Positive Impact:**
- Age 10s living population: 0.8325
- Attraction facilities count: 0.7688
- Average closure business duration: 0.6877
- Age 20s living population: 0.4971
- Bus stop count: 0.4380

**Lowest/Negative Impact:**
- University count: -0.8070
- Tuesday living population: -0.8006
- Age 40s living population: -0.5029
- Gentrification binary indicator: -0.4775
- Time slot 6 (20:00-24:00) population: -0.4754

### 6. 2023 Prediction Results
Applied the trained Transformer model to Q1 2023 data (342 commercial districts):

- **Cluster 0 (Warning Stage)**: 14 districts (4.1%)
- **Cluster 1 (Initial Stage)**: 176 districts (51.5%)
- **Cluster 2 (Risk Stage)**: 6 districts (1.8%)
- **Cluster 3 (Caution Stage)**: 146 districts (42.7%)

**High-Risk Districts (Stage 3 - Risk):**
- Omokgyo Station
- Guro Digital Complex
- Gangnam Eulji Hospital
- Eonju Station (Cha Hospital)
- Cheongdam Intersection (Cheongdam Luxury Street)
- Gaerong Station

## Key Findings

### Gentrification Diagnosis (2022)
<img width="286" height="138" alt="Screenshot 2025-10-08 at 12 45 16" src="https://github.com/user-attachments/assets/ebd093b0-3c2e-4f83-8f4f-e981d32d26e4" />

Visualization showing the actual gentrification status across districts in 2022, with color-coded severity levels based on observed data.

### Gentrification Prediction (2023)
<img width="246" height="162" alt="Screenshot 2025-10-08 at 12 45 02" src="https://github.com/user-attachments/assets/6f1e70fc-85f4-4ff3-84af-ab65764cf2dd" />

Transformer model prediction for 2023, demonstrating the model's capability to forecast gentrification patterns using time-series analysis.

### Critical Insights

**Primary Drivers:**
   
<img width="293" height="450" alt="Screenshot 2025-10-08 at 12 14 18" src="https://github.com/user-attachments/assets/a7792942-6854-49ce-9de9-20307e7f61bf" />

1. **Commercial District Type**: Traditional markets (R) and developed commercial areas (D) showed the strongest correlation with gentrification progression
2. **Education Expenditure**: High education spending indicates affluent population inflow, a key gentrification marker
4. **Youth Demographics**: High concentration of people in their 10s and 20s signals changing neighborhood character
5. **Infrastructure Density**: Attraction facilities and public transportation access accelerate gentrification


**Model Performance:**
- Deep learning (Transformer) significantly outperformed traditional machine learning methods
- 99% accuracy demonstrates robust pattern recognition capability
- Low RMSE (0.0641) indicates precise stage classification

**Policy Implications:**
- 51.5% of districts in initial stage require monitoring to prevent uncontrolled gentrification
- 6 high-risk districts need immediate intervention strategies
- Early warning system can help policymakers implement protective measures before displacement occurs

## Technologies Used

### Programming Languages
- Python 3.x
- SQL

### Machine Learning Frameworks
- Scikit-learn: Classical ML algorithms
- PyTorch: Deep learning model implementation
- PyCaret 3.2.0: Automated ML model selection and tuning

### Database
- MariaDB: Relational database for data storage and integration

### Data Processing Libraries
- Pandas: Data manipulation and analysis
- NumPy: Numerical computations
- Papaparse: CSV processing

### Visualization Tools
- Matplotlib: Static visualizations
- Seaborn: Statistical graphics
- PCA: Dimensionality reduction visualization

## Results Summary

### Model Comparison

| Model | Accuracy/RMSE | Key Strength |
|-------|---------------|--------------|
| Logistic Regression | 59.2% accuracy | Baseline model |
| Extra Trees Regressor | 0.0545 RMSE | Feature importance analysis |
| Random Forest Regressor | 0.0532 RMSE | Robust predictions |
| **Transformer (Final)** | **99% / 0.0641 RMSE** | **Best overall performance** |

### Research Contributions
1. **Data Integration**: Successfully unified multi-source urban datasets into a scalable relational database
2. **Diagnosis Framework**: Implemented and validated the government's gentrification stage classification system
3. **Predictive Capability**: Built an interpretable ML pipeline capable of forecasting gentrification risks with 99% accuracy
4. **Feature Analysis**: Identified traditional markets, developed commercial areas, and education expenditure as primary gentrification indicators
5. **Actionable Insights**: Provided spatial cluster visualization and district-level predictions for urban planning

### Limitations & Future Work
- **Data Availability**: Income bracket data unavailable for 2023; future research should incorporate complete demographic indicators
- **Time Series**: Current model uses quarterly snapshots; LSTM or temporal transformers could capture dynamic trends
- **External Factors**: Policy changes, COVID-19 impacts, and macro-economic conditions not fully integrated
- **Validation**: Longitudinal validation needed to assess prediction accuracy against actual 2023-2024 outcomes

## Team & Contributions

**Project Type**: Capstone Design Program  
**Institution**: Hankuk University of Foreign Studies  
**Department**: Artificial Intelligence Convergence – Business AI Track  
**Period**: Fall 2023

**Team Members:**
- Kim Seonmin (GBT, Class of 2020) - Project Leader, Database Design, Model Architecture
- Park Hwarang (Hungarian Studies, Class of 2019) - Data Collection, Feature Engineering
- Bang Eunseon (Japanese Language & Culture, Class of 2019) - Data Analysis, Visualization
- Choi Yeonjae (English Interpretation & Translation, Class of 2020) - Machine Learning Implementation

**Project Leader Responsibilities:**
- Database schema design and SQL-based data integration
- Transformer model architecture design and hyperparameter optimization
- Attention mechanism analysis and feature importance interpretation
- Spatial clustering visualization and predictive analytics

## Citations

This research builds upon:
- Ministry of Land, Infrastructure and Transport (2020). Gentrification Diagnosis System
- Lee et al. (2019). Gentrification Indicator Development and Application
- Kim & Park (2023). Place Identity Analysis of Gentrified Areas Using Big Data

## Acknowledgments

This project was conducted as part of the Capstone Design Program at Hankuk University of Foreign Studies, Department of Artificial Intelligence Convergence – Business AI Track (Fall 2023). We thank the Seoul Metropolitan Government for providing open access to commercial district data through the Seoul Open Data Plaza.


---
## Notes from my refactor

# Gentrification Prediction Capstone

Refactored toolkit for experimenting with gentrification prediction models. The project now exposes reusable Python modules plus simple CLI scripts for training a transformer classifier and running baseline classical ML demos.

## Repository Layout

- `src/gentrification/` – reusable code for data loading, model definition, training utilities, inference helpers, visualisation, and classic ML demos.
- `scripts/` – command line entry points (`train_transformer.py`, `predict_transformer.py`).
- `notebooks/` – original research notebooks preserved for reference.
- `artifacts/` – default output directory for checkpoints and preprocessing artefacts (created after training).

## Quick Start

1. Install dependencies (example using pip):
   ```bash
   pip install torch scikit-learn pandas numpy matplotlib seaborn joblib
   ```
2. Train the transformer model on your CSV dataset:
   ```bash
   python scripts/train_transformer.py path/to/result_after_cluster.csv --target clust --feature-start 3
   ```
   Adjust `--drop-columns` or `--feature-columns` when your schema differs.
3. Generate predictions for new data using the saved artefacts:
   ```bash
   python scripts/predict_transformer.py path/to/newdata2023.csv --artifacts artifacts --output predictions.csv
   ```

## Module Overview

- `gentrification.data` – `DataConfig` for declaring dataset schema and `prepare_datasets` for building PyTorch dataloaders together with scalers and label encoders.
- `gentrification.model` – `TransformerClassifier` exposes a batch-first multi-head attention model tailored for tabular data.
- `gentrification.training` – `TrainingConfig` and `train_model` implement early-stopped optimisation with tracked metrics.
- `gentrification.prediction` – utilities for loading checkpoints and producing batched predictions.
- `gentrification.visualization` – helpers to convert attention weights into dataframes and bar charts.
- `gentrification.demos` – reusable Iris/MNIST demo functions covering KNN, KMeans, and decision trees.

## Next Steps

- Extend `scripts/train_transformer.py` with custom logging or wandb integration when experimenting at scale.
- Wrap data preprocessing (categorical encoding, imputation) before calling `prepare_datasets` if your inputs require it.

