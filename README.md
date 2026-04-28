# Predicting-Municipal-Housing-Prices---Machine-Learning

## 1. Overview
This project builds upon a previous data integration effort using public datasets from PORDATA, in which multiple socioeconomic indicators were collected, cleaned, and consolidated at the municipal level.
The objective of the current project was to apply supervised machine learning techniques to predict the median housing price per square meter across Portuguese municipalities. Specifically, the project was aimed at practicing the full machine learning workflow, including feature engineering, model training, evaluation, comparison, and interpretation. 
The methodology follows concepts presented in Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido.

## 2. Data Sources
The dataset used in this project is the consolidated municipal-level dataset generated in the previous data integration project (2024 data only), which combines multiple socioeconomic indicators obtained from the PORDATA municipal database:
- Resident population  
- Crude mortality rate 
- Infant mortality rate 
- Crude birth rate 
- Unemployment (total)
- Average monthly income 
- Number of pharmacies
- Total crimes
- Crude nuptiality rate
- Crude divorce rate
A new target variable (median housing price per square meter) was added to the consolidated dataset to enable supervised learning. 

## 3. Feature Engineering
Additional derived variables were created to improve comparability across municipalities and enhance model performance: 
- Population balance = Birth rate - Mortality rate
- Crimes per 1.000 inhabitants
- Unemployment rate (%)
- Pharmacies per 10.000 inhabitants
These variables were constructed by normalizing absolute counts by population, allowing municipalities of different sizes to be compared on a common scale. 
After normalization, the original absolute-count variables were removed to avoid scale effects and reduce redundancy. 

## 4. Supervised Learning Workflow
The project followed a standard supervised learning workflow inspired by Introduction to Machine Learning with Python.

### 4.1 Feature and Target Definition
- The target variable was defined as the median housing price per square meter
- All remaining socioeconomic indicators were used as features
- Non-predictive columns such as Municipality and Year were removed
- Observations with missing target values were excluded

### 4.2 Train-Test Split
- The dataset was split into training and testing subsets
- A 75% / 25% split was used
- The split was performed before preprocessing to avoid data leakage
- A fixed random_state (42) was used to ensure reproducibility

### 4.3 Data Preprocessing
Preprocessing steps were applied using scikit-learn pipelines:
- Missing values were imputed using SimpleImputer (mean strategy)
- Numerical features were standardized using StandardScaler (mean = 0, standard deviation = 1)
- Preprocessing transformations were fitted only on the training data and applied to the test data via pipelines
Feature scaling was particularly important for linear and regularized models, as variables operate on different numerical scales. 

### 4.4 Models Implemented
Three regression models were trained and compared: 
- Linear Regression (baseline model) 
- Ridge Regression
- Random Forest Regressor

### 4.5 Model Evaluation
Models were evaluated using: 
- R² score on training and test sets
- 5-fold cross-validation (R²)
- Coefficient analysis (linear models) 
- Feature importance analysis (Random Forest)
This allowed to compare model performance and generalization ability. 

## 5. Model Comparison
| Model            | Train R² | Test R² | CV Mean R² |
|------------------|----------|---------|------------|
| Linear Regression| ~0.67    | ~0.78   | ~0.65      |
| Ridge Regression | ~0.67    | ~0.79   | ~0.65      |
| Random Forest    | ~0.95    | ~0.74   | ~0.63      |

Linear models showed more stable generalization performance, while the Random Forest achieved higher training performance but exhibited mild overfitting.

## 6. Model Interpretability
### 6.1 Linear Regression Coefficients 
Linear model coefficients provide a direct and interpretable measure of the relationship between each feature and housing prices. Since features were standardized, coefficients represent the expected change in housing prices associated with a one standard deviation increase in each predictor.
Key findings:
- Population is the strongest positive predictor (+252), indicating a strong association between urban scale and higher housing prices
- Crimes per 1.000 inhabitants also shows a strong positive association (+212), likely reflecting urban density effects rather than causality 
- Mortality rate (−104) and unemployment rate (−65) are among the strongest negative predictors
- Average income (+45) has a positive but moderate effect compared to demographic variables

These results suggest that housing prices are primarily driven by urban scale and demographic structure, rather than purely economic variables. In particular, population-related variables dominate the model, reinforcing the importance of municipality size and urbanization effects in explaining housing price variation.

### 6.2 Random Forest Feature Importance
Feature importance from the Random Forest model provides a non-linear perspective on predictive relevance. The most important variables were:
1.	Population balance (0.43)
2.	Population (0.14)
3.	Crimes per 1.000 inhabitants (0.10)
4.	Mortality rate (0.07)
5.	Average income (0.05)
These results are broadly consistent with the linear model, reinforcing the importance of demographic structure and urban scale.

## 7. Sensitivity Analysis: Removing Population
To assess whether model performance was primarily driven by municipality size, the Population variable was removed and models were retrained.

| Model            | Train R² | Test R² | 
|------------------|----------|---------|
| Linear Regression| ~0.59    | ~0.69   | 
| Ridge Regression | ~0.59    | ~0.69   | 
| Random Forest    | ~0.94    | ~0.67   | 

Performance decreased moderately, indicating that population contributes to predictive accuracy but is not the sole driver. Socioeconomic indicators retain meaningful predictive power.

## 8. Key Findings
- Housing prices per square meter are moderately explained by municipal socio-economic indicators (R² of approximately 0.78 in linear models).
- Linear models showed better generalization performance than the Random Forest, which exhibited signs of overfitting.
- Demographic variables, particularly population and population growth balance, are among the main predictors of housing prices.
- Removing the population variable led to a moderate decrease in model performance, indicating that urban scale is relevant but not the sole driver of housing price variation.

## 9. Tools and Technologies
- Python
- pandas
- scikit-learn
- matplotlib
- Jupyter Notebook

## 10. Repository Structure
```
data/
  processed/   # Consolidated dataset from previous project
notebooks/
  01_modeling.ipynb   # Machine learning workflow and analysis
README.md
```

## 11. How to Reproduce
1. Clone the repository.
2. Place the consolidated dataset in data/processed/
3. Open 01_modeling.ipynb
4. Run all cells
