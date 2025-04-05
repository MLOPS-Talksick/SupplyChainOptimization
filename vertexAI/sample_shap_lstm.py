

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import shap
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv("E:/MLOps/SupplyChainOptimization/filename.csv")

# Convert data types
df['Total_Quantity'] = df['Total_Quantity'].astype(int)
df['Date'] = pd.to_datetime(df['Date']) # , format='%d-%m-%Y')

# Sort data by date
df = df.sort_values('Date')

# Extract date features
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek
df['dayofyear'] = df['Date'].dt.dayofyear
df['quarter'] = df['Date'].dt.quarter

# Add more sample data to demonstrate time series modeling
# Let's create 100 days of data for each product
dates = pd.date_range(start='2023-01-01', periods=100)
products = df['Product_Name'].unique()

# Create expanded DataFrame
expanded_data = []
for date in dates:
    for product in products:
        # Create some pattern: base value + day of week effect + trend + noise
        base_value = np.random.randint(10, 40)
        day_effect = date.dayofweek * 2  # Higher sales on weekends
        trend = date.dayofyear * 0.1  # Slight upward trend over time
        seasonality = 5 * np.sin(date.dayofyear / 30 * np.pi)  # Monthly seasonality
        noise = np.random.normal(0, 3)  # Random noise
        
        quantity = max(1, int(base_value + day_effect + trend + seasonality + noise))
        
        expanded_data.append({
            'Date': date,
            'Product_Name': product,
            'Total_Quantity': quantity
        })

df_expanded = pd.DataFrame(expanded_data)

# Extract the same features for expanded data
df_expanded['year'] = df_expanded['Date'].dt.year
df_expanded['month'] = df_expanded['Date'].dt.month
df_expanded['day'] = df_expanded['Date'].dt.day
df_expanded['dayofweek'] = df_expanded['Date'].dt.dayofweek
df_expanded['dayofyear'] = df_expanded['Date'].dt.dayofyear
df_expanded['quarter'] = df_expanded['Date'].dt.quarter
# df_expanded['product_id'] = df_expanded['Product_Name'].str.extract('(\d+)').astype(int)

# Calculate additional features that might help prediction
# 1. Rolling statistics for each product
df_expanded = df_expanded.sort_values(['Product_Name', 'Date'])
df_expanded['rolling_mean_7d'] = df_expanded.groupby('Product_Name')['Total_Quantity'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean())
df_expanded['rolling_std_7d'] = df_expanded.groupby('Product_Name')['Total_Quantity'].transform(
    lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))

# 2. Lag features
for lag in [1, 2, 3, 7]:  # 1-day, 2-day, 3-day, and 7-day lags
    df_expanded[f'lag_{lag}d'] = df_expanded.groupby('Product_Name')['Total_Quantity'].shift(lag)

# Fill NaN values
df_expanded = df_expanded.fillna(0)

# Encode product names
label_encoder = LabelEncoder()
df_expanded['product_encoded'] = label_encoder.fit_transform(df_expanded['Product_Name'])

# Normalize numeric features
features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter', 
            # 'product_id', 
            'product_encoded', 'rolling_mean_7d', 'rolling_std_7d',
            'lag_1d', 'lag_2d', 'lag_3d', 'lag_7d']

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df_expanded[features])
y_scaled = scaler_y.fit_transform(df_expanded[['Total_Quantity']])

# Prepare data for LSTM (time steps)
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Group by product_encoded, then create sequences for each product
all_X_seq = []
all_y_seq = []
time_steps = 5

for product in df_expanded['product_encoded'].unique():
    product_indices = df_expanded[df_expanded['product_encoded'] == product].index
    product_X = X_scaled[product_indices]
    product_y = y_scaled[product_indices]
    
    if len(product_X) > 15:  # Ensure we have enough data points for sequences
        X_seq, y_seq = create_sequences(product_X, product_y, time_steps=time_steps)
        all_X_seq.append(X_seq)
        all_y_seq.append(y_seq)

# Combine all sequences
X_seq = np.vstack(all_X_seq)
y_seq = np.vstack(all_y_seq)

# Save the feature names for SHAP explanation
feature_names = features.copy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Train Loss: {train_loss:.4f}')
print(f'Test Loss: {test_loss:.4f}')

# Make predictions
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_test_actual) ** 2))
print(f'RMSE: {rmse:.4f}')

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot predictions vs actual for a sample of test data
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.title('Predicted vs Actual Quantities')
plt.ylabel('Quantity')
plt.xlabel('Sample Index')
plt.legend()
plt.show()

###################################
# Alternative to SHAP for LSTM Model
###################################

# Create a feature importance analyzer for LSTM models
class LSTMFeatureImportance:
    def __init__(self, model, X_data, feature_names, time_steps):
        self.model = model
        self.X_data = X_data
        self.feature_names = feature_names
        self.time_steps = time_steps
        self.predictions = model.predict(X_data)
        
    def permutation_importance(self, num_repeats=10):
        """
        Calculate feature importance using permutation importance.
        This method shuffles each feature and measures the change in prediction error.
        """
        # Baseline score
        baseline_pred = self.model.predict(self.X_data)
        baseline_error = np.mean((baseline_pred - self.model.predict(self.X_data)) ** 2)
        
        # Initialize importance scores
        importances = np.zeros((len(self.feature_names), self.time_steps))
        
        # For each feature and time step
        for feat_idx, feature in enumerate(self.feature_names):
            for t in range(self.time_steps):
                importance_samples = []
                
                for _ in range(num_repeats):
                    # Create a copy of the data
                    X_permuted = self.X_data.copy()
                    
                    # Shuffle the values for this feature at this time step
                    np.random.shuffle(X_permuted[:, t, feat_idx])
                    
                    # Predict with the permuted feature
                    perm_pred = self.model.predict(X_permuted)
                    
                    # Calculate permutation error
                    perm_error = np.mean((perm_pred - baseline_pred) ** 2)
                    
                    # The importance is the increase in error
                    importance = perm_error - baseline_error
                    importance_samples.append(importance)
                
                # Average importance across repeats
                importances[feat_idx, t] = np.mean(importance_samples)
        
        return importances
    
    def create_feature_importance_df(self, importances):
        """
        Create a DataFrame with feature importance results.
        """
        # Flatten importances
        feature_time_importance = []
        
        for feat_idx, feature in enumerate(self.feature_names):
            for t in range(self.time_steps):
                time_step_label = f"t-{self.time_steps-t}"
                feature_time_importance.append({
                    'Feature': feature,
                    'Time_Step': time_step_label,
                    'Importance': importances[feat_idx, t]
                })
        
        importance_df = pd.DataFrame(feature_time_importance)
        
        # Also create a grouped importance by feature
        grouped_importance = importance_df.groupby('Feature')['Importance'].sum().reset_index()
        grouped_importance = grouped_importance.sort_values('Importance', ascending=False)
        
        return importance_df, grouped_importance
    
    def plot_importance(self, importances):
        """
        Plot feature importance.
        """
        # Create dataframes
        importance_df, grouped_importance = self.create_feature_importance_df(importances)
        
        # Plot grouped importance
        plt.figure(figsize=(12, 8))
        plt.barh(grouped_importance['Feature'], grouped_importance['Importance'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance (Summed across time steps)')
        plt.tight_layout()
        plt.show()
        
        # Plot feature importance by time step (top features only)
        top_features = grouped_importance['Feature'].head(8).tolist()
        time_importance = importance_df[importance_df['Feature'].isin(top_features)]
        
        plt.figure(figsize=(14, 10))
        for feature in top_features:
            feature_data = time_importance[time_importance['Feature'] == feature]
            plt.plot(feature_data['Time_Step'], feature_data['Importance'], marker='o', label=feature)
        
        plt.xlabel('Time Step')
        plt.ylabel('Importance')
        plt.title('Feature Importance by Time Step')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return importance_df, grouped_importance

# Run permutation importance
print("\nCalculating feature importance using permutation importance...")
feature_importance_analyzer = LSTMFeatureImportance(model, X_test[:100], feature_names, time_steps)
importances = feature_importance_analyzer.permutation_importance(num_repeats=5)
importance_by_time, importance_grouped = feature_importance_analyzer.plot_importance(importances)

print("\nTop 10 Features by Overall Importance:")
print(importance_grouped.head(10))

# Function to analyze feature importance for specific products
def analyze_product_importance(product_name, num_samples=50):
    product_idx = label_encoder.transform([product_name])[0]
    product_data = df_expanded[df_expanded['product_encoded'] == product_idx].sort_values('Date')
    
    # Create sequences for this product
    product_indices = product_data.index
    product_X = X_scaled[product_indices]
    product_y = y_scaled[product_indices]
    
    if len(product_X) <= time_steps:
        return f"Not enough data for product {product_name}"
    
    X_seq_product, y_seq_product = create_sequences(product_X, product_y, time_steps=time_steps)
    
    # Use only a subset for analysis
    if len(X_seq_product) > num_samples:
        X_seq_product = X_seq_product[:num_samples]
    
    # Run permutation importance for this product
    product_analyzer = LSTMFeatureImportance(model, X_seq_product, feature_names, time_steps)
    product_importances = product_analyzer.permutation_importance(num_repeats=3)
    _, product_importance_grouped = product_analyzer.create_feature_importance_df(product_importances)
    
    print(f"\nTop features for predicting {product_name}:")
    print(product_importance_grouped.head(10))
    
    # Plot importance
    plt.figure(figsize=(10, 6))
    plt.barh(product_importance_grouped['Feature'].head(10), 
             product_importance_grouped['Importance'].head(10))
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance for {product_name}')
    plt.tight_layout()
    plt.show()
    
    return product_importance_grouped

# Analyze a specific product
sample_product = products[0]
product_importance = analyze_product_importance(sample_product)

###################################
# Feature Contribution Analysis
###################################

def analyze_feature_contribution(model, X_sample, feature_names, time_steps=5):
    """
    Analyze how individual features contribute to predictions by varying their values.
    """
    # Get baseline prediction
    baseline_pred = model.predict(X_sample)[0][0]
    
    # Initialize contribution results
    contributions = {}
    
    # For each feature
    for feat_idx, feature in enumerate(feature_names):
        # Test different values for this feature
        test_values = np.linspace(0, 1, 10)  # Try 10 different values
        predictions = []
        
        for value in test_values:
            # Create a copy of the data
            X_modified = X_sample.copy()
            
            # Set all time steps for this feature to the test value
            for t in range(time_steps):
                X_modified[0, t, feat_idx] = value
            
            # Get prediction with modified feature
            pred = model.predict(X_modified)[0][0]
            predictions.append(pred)
        
        # Calculate slope of prediction change
        slope = np.polyfit(test_values, predictions, 1)[0]
        
        # Store contribution
        contributions[feature] = {
            'slope': slope,
            'values': test_values,
            'predictions': predictions
        }
    
    # Sort contributions by absolute slope
    sorted_contributions = sorted(contributions.items(), 
                                  key=lambda x: abs(x[1]['slope']), 
                                  reverse=True)
    
    # Plot top contributing features
    plt.figure(figsize=(14, 8))
    for i, (feature, data) in enumerate(sorted_contributions[:6]):  # Top 6 features
        plt.subplot(2, 3, i+1)
        plt.plot(data['values'], data['predictions'])
        plt.title(f"{feature} (slope: {data['slope']:.2f})")
        plt.xlabel('Feature Value')
        plt.ylabel('Prediction')
        plt.axhline(y=baseline_pred, color='r', linestyle='--', alpha=0.5)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return sorted_contributions

# Analyze feature contribution for a sample
print("\nAnalyzing how features contribute to predictions...")
sample_idx = np.random.randint(0, len(X_test))
feature_contributions = analyze_feature_contribution(model, X_test[sample_idx:sample_idx+1], feature_names)

print("\nTop features by contribution magnitude:")
for feature, data in feature_contributions[:10]:
    print(f"{feature}: slope = {data['slope']:.4f}")

###################################
# Additional analysis for specific time periods
###################################

def analyze_seasonal_patterns():
    """
    Analyze how day of week and other seasonal factors affect predictions.
    """
    # Group data by day of week
    dow_groups = df_expanded.groupby('dayofweek')['Total_Quantity'].mean()
    
    plt.figure(figsize=(10, 6))
    plt.bar(dow_groups.index, dow_groups.values)
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Average Quantity')
    plt.title('Average Quantity by Day of Week')
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.grid(True, axis='y')
    plt.show()
    
    # Group by month
    df_expanded['month_name'] = df_expanded['Date'].dt.month_name()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    month_groups = df_expanded.groupby('month_name')['Total_Quantity'].mean()
    month_groups = month_groups.reindex(month_order)
    
    plt.figure(figsize=(12, 6))
    plt.bar(month_groups.index, month_groups.values)
    plt.xlabel('Month')
    plt.ylabel('Average Quantity')
    plt.title('Average Quantity by Month')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Get feature importance for specific time periods
    weekday_data = df_expanded[df_expanded['dayofweek'] < 5]
    weekend_data = df_expanded[df_expanded['dayofweek'] >= 5]
    
    print("\nKey features on weekdays vs weekends:")
    print("Weekday data shape:", weekday_data.shape)
    print("Weekend data shape:", weekend_data.shape)

# Analyze seasonal patterns
analyze_seasonal_patterns()

# Function to predict future quantities for a specific product
def predict_future_quantities(product_name, days=30):
    # Get the last sequence for the product
    product_idx = label_encoder.transform([product_name])[0]
    product_data = df_expanded[df_expanded['product_encoded'] == product_idx].sort_values('Date')
    
    if len(product_data) < 5:  # Need at least time_steps data points
        return "Not enough historical data for this product"
    
    last_date = product_data['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    
    # Extract the last known sequence
    last_sequence = X_scaled[product_data.index[-5:]]
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    for i in range(days):
        # Reshape for prediction
        current_sequence_reshaped = current_sequence.reshape(1, 5, -1)
        
        # Predict next day
        next_pred_scaled = model.predict(current_sequence_reshaped)
        next_pred = scaler_y.inverse_transform(next_pred_scaled)[0][0]
        predictions.append(next_pred)
        
        # Get feature values for the next day
        next_date = future_dates[i]
        next_features = np.zeros(len(features))
        
        # Update date features
        next_features[0] = next_date.year
        next_features[1] = next_date.month
        next_features[2] = next_date.day
        next_features[3] = next_date.dayofweek
        next_features[4] = next_date.dayofyear
        next_features[5] = next_date.quarter
        
        # Product features remain the same
        next_features[6] = product_idx  # Moved from index 7

        # Update lag features based on predictions
        if i == 0:
            # For the first prediction, use values from the dataset
            next_features[7] = product_data['rolling_mean_7d'].iloc[-1]  # rolling_mean_7d
            next_features[8] = product_data['rolling_std_7d'].iloc[-1]   # rolling_std_7d
            next_features[9] = product_data['Total_Quantity'].iloc[-1]   # lag_1d
            next_features[10] = product_data['Total_Quantity'].iloc[-2] if len(product_data) > 1 else 0  # lag_2d
            next_features[11] = product_data['Total_Quantity'].iloc[-3] if len(product_data) > 2 else 0  # lag_3d
            next_features[12] = product_data['Total_Quantity'].iloc[-7] if len(product_data) > 6 else 0  # lag_7d
        else:
            # For subsequent predictions, use the predicted values
            if i >= 7:
                next_features[7] = np.mean(predictions[i-7:i])  # rolling_mean_7d
                next_features[8] = np.std(predictions[i-7:i]) if len(predictions[i-7:i]) > 1 else 0  # rolling_std_7d
            else:
                # For the first few days, use a mix of historical and predicted
                historical = list(product_data['Total_Quantity'].iloc[-(7-i):])
                predicted = predictions[:i]
                combined = historical + predicted
                next_features[7] = np.mean(combined)  # rolling_mean_7d
                next_features[8] = np.std(combined) if len(combined) > 1 else 0  # rolling_std_7d

            next_features[9] = predictions[i-1]  # lag_1d
            next_features[10] = predictions[i-2] if i >= 2 else product_data['Total_Quantity'].iloc[-1]  # lag_2d
            next_features[11] = predictions[i-3] if i >= 3 else product_data['Total_Quantity'].iloc[-2]  # lag_3d
            next_features[12] = predictions[i-7] if i >= 7 else product_data['Total_Quantity'].iloc[-6]  # lag_7d
        
        # Scale the features
        next_features_scaled = scaler_X.transform(next_features.reshape(1, -1))
        
        # Update the sequence for the next iteration (remove first, add latest)
        current_sequence = np.vstack([current_sequence[1:], next_features_scaled])
    
    # Create a DataFrame with the predictions
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Product_Name': product_name,
        'Predicted_Quantity': [max(1, int(round(pred))) for pred in predictions]
    })
    
    return future_df

# Example: Predict next 30 days for one product
sample_product = products[0]
future_predictions = predict_future_quantities(sample_product, days=30)
print(f"\nFuture predictions for {sample_product}:")
print(future_predictions)

# Function to predict for all products
def predict_all_products(days=30):
    all_predictions = []
    for product in products:
        pred_df = predict_future_quantities(product, days)
        if isinstance(pred_df, pd.DataFrame):
            all_predictions.append(pred_df)
    
    if all_predictions:
        return pd.concat(all_predictions, ignore_index=True)
    else:
        return "No predictions available"