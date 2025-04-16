# Demand Forecasting Model Bias Assessment Report
## Generated on: 2025-04-15 10:16:21

## 1. Data Distribution Analysis

### 1.1 Product Distribution in Training Data
- Total number of products: 66
- Most frequent product: beef (2190 samples)
- Least frequent product: varun (1 samples)
- Imbalance ratio (max/min): 2190.00

⚠️ **Warning**: Significant class imbalance detected. Consider data augmentation or weighted training.

### 1.2 Time Coverage Analysis
- Minimum time coverage: 0 days
- Maximum time coverage: 2189 days
- Coverage ratio (max/min): 2189.00

⚠️ **Warning**: Significant variation in time coverage between products. Some products may lack sufficient historical data.

### 1.3 Outlier Analysis
- Maximum outlier percentage: 34.88%
- Average outlier percentage: 5.69%

⚠️ **Warning**: High percentage of outliers detected. Consider robust scaling or outlier removal.

## 2. Temporal Bias Analysis

### 2.1 Autocorrelation Analysis
- Maximum weekly autocorrelation: 0.65
- Minimum weekly autocorrelation: -0.33

### 2.2 Time Gaps Analysis
- Maximum gap in data: 110.0 days

⚠️ **Warning**: Significant gaps in time series data detected. Consider imputation for missing periods.

### 2.3 Recency Bias Analysis
- Average ratio of data from last month: 0.02

## 3. Prediction Bias Analysis
- Mean prediction error: -4.65
- Median prediction error: 2.36
- Correlation between error and actual value: -0.43
- Heteroscedasticity correlation: 0.34 (p-value: 0.0000)

⚠️ **Warning**: Correlation between error and actual value detected. Model performance varies with demand magnitude.

⚠️ **Warning**: Heteroscedasticity detected. Error variance increases with demand magnitude.

## 4. Model Fairness Analysis

### 4.1 Overall Model Performance
- Overall RMSE: 43.37
- Overall MAE: 19.97
- Overall MAPE: 297.42%

### 4.2 Performance Across Products
- RMSE disparity ratio (max/min): 1.16
- MAPE disparity ratio (max/min): 9.52
- Worst RMSE performance: baking goods (RMSE: 42.89)
- Worst MAPE performance: baking goods (MAPE: 87.37%)

## 5. Recommendations for Bias Mitigation
- **Data Augmentation**: Apply data augmentation techniques for products with low sample counts.
- **Weighted Loss Function**: Implement weighted loss function to give more importance to underrepresented products.
- **Historical Data Collection**: Collect more historical data for products with limited time coverage.
- **Transfer Learning**: Use knowledge from products with extensive history to inform predictions for newer products.
- **Robust Scaling**: Use robust scaling methods (like RobustScaler) that are less sensitive to outliers.
- **Anomaly Detection**: Implement anomaly detection to identify and handle extreme values appropriately.
- **Time Series Imputation**: Apply time series imputation techniques to fill gaps in the data.
- **Log Transformation**: Apply log transformation to the target variable if errors increase with demand magnitude.