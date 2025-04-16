# Debiased Demand Forecasting Model Report
## Generated on: 2025-04-15 11:19:11

## 1. Data Summary
- Original dataset: 36414 samples
- Cleaned dataset: 36414 samples
- Balanced dataset: 36501 samples
- Number of products: 66
- Date range: 2019-01-03 to 2025-12-18

## 2. Bias Mitigation Techniques Applied
- **Outlier Handling**: Used winsorization to handle extreme values
- **Data Balancing**: Generated synthetic samples for underrepresented products
- **Robust Scaling**: Used RobustScaler instead of standard scaling
- **Log Transformation**: Applied to target variable to handle heteroscedasticity
- **Weighted Training**: Used inverse frequency weighting during model training
- **Fairness-Aware Training**: Used custom callback to monitor performance across products
- **Bias Correction**: Applied post-processing correction to remove systematic bias

## 3. Model Architecture
- LSTM layers: 2 (units: 64, 32)
- Activation functions: tanh, tanh
- Dropout rates: 0.2, 0.2
- Dense layer units: 16
- Optimizer: adam (learning rate: 0.001)

## 4. Model Performance
### Overall Performance
- RMSE: 16.12
- MAE: 9.20
- MAPE: 37.10%

### Fairness Metrics
- RMSE disparity ratio: 1.00
- MAPE disparity ratio: 1.00
- Best performing product (RMSE): baking goods
- Worst performing product (RMSE): baking goods

## 5. Future Predictions
- Forecast horizon: 7 days
- Number of products forecast: 66
- Total predicted quantity: 1147 units

### Top 5 Products by Predicted Quantity
- wheat: 193 units
- milk: 170 units
- coffee: 111 units
- chocolate: 80 units
- beef: 70 units