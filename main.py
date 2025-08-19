import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
# --- 1. Synthetic Data Generation ---
# Generate synthetic user engagement data
np.random.seed(42)  # for reproducibility
num_users = 100
num_days = 365
dates = pd.date_range(start='2022-01-01', periods=num_days)
users = np.random.choice(range(1, num_users + 1), size=num_days * 10, replace=True)
daily_engagement = np.random.randint(1, 100, size=num_days * 10)
df = pd.DataFrame({'Date': np.repeat(dates, 10), 'User_ID': users, 'Daily_Engagement': daily_engagement})
df = df.groupby(['Date', 'User_ID'])['Daily_Engagement'].sum().reset_index()
# --- 2. Data Aggregation and Preparation ---
# Aggregate daily engagement to weekly engagement per user
df['Week'] = df['Date'].dt.isocalendar().week
weekly_engagement = df.groupby(['User_ID', 'Week'])['Daily_Engagement'].sum().reset_index()
#Example of a single user's time series
user_id_example = 1
user_data = weekly_engagement[weekly_engagement['User_ID'] == user_id_example].set_index('Week')['Daily_Engagement']
# --- 3. Time Series Analysis (Example for a single user) ---
# Fit an ARIMA model (example: ARIMA(1,1,1)) -  You would need to optimize the order for your data
try:
    model = ARIMA(user_data, order=(1,1,1))
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(user_data), end=len(user_data) + 4) #Predict next 4 weeks
    # --- 4. Visualization (Example for a single user) ---
    plt.figure(figsize=(10, 6))
    plt.plot(user_data, label='Observed')
    plt.plot(predictions, label='Predicted', linestyle='--')
    plt.xlabel('Week')
    plt.ylabel('Weekly Engagement')
    plt.title(f'Weekly Engagement for User {user_id_example} - ARIMA(1,1,1) Prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot to a file
    output_filename = 'user_engagement_prediction.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
except Exception as e:
    print(f"Error fitting ARIMA model: {e}")
#Note:  This is a simplified example.  A real-world application would require:
#       - More robust data cleaning and preprocessing.
#       -  Feature engineering (e.g., adding user demographics).
#       -  Model selection and hyperparameter tuning (using techniques like cross-validation).
#       -  Evaluation metrics (e.g., RMSE, MAE) for model performance.
#       -  Iteration across all users.
#       -  Churn prediction based on the model output (e.g., setting a threshold for low engagement).