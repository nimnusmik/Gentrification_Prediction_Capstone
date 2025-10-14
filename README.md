# Gentrification Prediction in Seoul Using Transformer-Based Deep Learning


## Overview
This project aims to analyze and predict gentrification patterns in Seoul’s commercial districts using multidimensional socioeconomic and spatial data.  
By integrating various datasets into a MariaDB-based infrastructure, the team identified urban transformation trends and built a predictive model leveraging both clustering and deep learning techniques.

## Objectives
- Detect patterns of urban gentrification based on business district data.  
- Identify regional clusters showing early signs of socioeconomic transformation.  
- Develop a data-driven predictive framework for future gentrification risk analysis.

## Data Description
- **Source**: Seoul Open Data Plaza, Ministry of Land and Transport, and regional business registries.  
- **Data Types**:
  - Commercial rent and business turnover rates.  
  - Population and income distribution by district.  
  - Building usage, zoning, and infrastructure data.  
- **Database**: All raw data were normalized and stored in **MariaDB**, ensuring efficient querying and integration.

## Methodology
1. **Data Integration & Preprocessing**
   - Combined heterogeneous data (CSV, JSON, API) into a unified relational schema using SQL.
   - Handled missing and noisy entries through statistical imputation and normalization.
2. **Clustering Analysis**
   - Applied **K-Means clustering** to categorize commercial districts based on socioeconomic indicators.
   - Determined the optimal number of clusters via the **Elbow Method**.
3. **Model Development**
   - Designed a **Transformer-based regression model** to predict future gentrification scores.
   - Tuned hyperparameters through grid search to optimize performance.
4. **Evaluation**
   - Achieved **98% accuracy** on validation data.
   - Conducted cross-validation to ensure robustness and minimize overfitting.

## Key Findings

### Gentrification Diagnosis (2022)
<img width="286" height="138" alt="Screenshot 2025-10-08 at 12 45 16" src="https://github.com/user-attachments/assets/ebd093b0-3c2e-4f83-8f4f-e981d32d26e4" />

Visualization showing the actual gentrification status across districts in 2022, with color-coded severity levels based on observed data.

### Gentrification Prediction (2023)
<img width="246" height="162" alt="Screenshot 2025-10-08 at 12 45 02" src="https://github.com/user-attachments/assets/6f1e70fc-85f4-4ff3-84af-ab65764cf2dd" />

Transformer model prediction for the following year (2023), demonstrating the model's capability to forecast gentrification patterns using time-series analysis.

### Analysis Results
- **Key Drivers Identified**: Rent increase rates, franchise store inflow, and demographic shifts emerged as primary gentrification factors
- **High-Risk Districts**: Successfully highlighted vulnerable areas requiring early policy intervention
- **Model Performance**: Deep learning models demonstrated superior pattern recognition and predictive power compared to traditional regression methods

## Technologies Used
- **Languages**: Python, SQL  
- **Frameworks**: PyTorch, Scikit-learn  
- **Database**: MariaDB  
- **Visualization**: Matplotlib, Seaborn  
- **Libraries**: Pandas, Numpy  

## Results
- Successfully integrated multi-source urban data into a scalable relational database.  
- Built an interpretable ML pipeline capable of forecasting gentrification risks in Seoul up to 2030.  
- Provided actionable insights for urban planners through spatial cluster visualization and predictive analytics.

## Team & Role
- **Role**: Project Leader / Data Scientist  
- **Responsibilities**:
  - Database design and SQL-based data integration.  
  - Model architecture design and hyperparameter optimization.  
  - Visualization of clustering and predictive outcomes.

## Acknowledgment
This project was conducted as part of the **Capstone Design Program** at Hankuk University of Foreign Studies,  
Department of Artificial Intelligence Convergence – Business AI Track (Fall 2023).
