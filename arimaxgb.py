
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

file_path = "coir_sales_dataset_2019_2025.csv"
df = pd.read_csv(file_path)

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter
df['Month'] = df['Date'].dt.month

print("\nMissing Values Before Handling:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

df = df.dropna()
df = df.drop_duplicates()

le = LabelEncoder()
df['Product_Encoded'] = le.fit_transform(df['Product'])

print("\nEncoded Products Mapping:")
for i, label in enumerate(le.classes_):
    print(label, "->", i)

Q1 = df['Quantity_Sold'].quantile(0.25)
Q3 = df['Quantity_Sold'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['Quantity_Sold'] >= lower) & (df['Quantity_Sold'] <= upper)]

print("\nDataset Shape After Outlier Removal:", df.shape)


plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(12,6))
for product in df['Product'].unique():
    temp = df[df['Product'] == product]
    plt.plot(temp['Date'], temp['Quantity_Sold'], label=product)

plt.title("Sales Trend by Product")
plt.xlabel("Date")
plt.ylabel("Quantity Sold")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Rainfall_mm'], y=df['Quantity_Sold'])
plt.title("Rainfall vs Sales")
plt.show()

products = df['Product'].unique()

for product_name in products:

    print("\n" + "="*60)
    print(f"Processing Product: {product_name}")
    print("="*60)

    df_product = df[df['Product'] == product_name].copy()
    df_product = df_product.set_index('Date')

    df_monthly = df_product.resample('MS').agg({
        'Quantity_Sold': 'sum',
        'Price': 'mean',
        'Festival_Indicator': 'mean',
        'Rainfall_mm': 'mean',
        'Raw_Material_Cost_per_kg': 'mean',
        'Production_Volume': 'sum'
    })

    df_monthly = df_monthly.dropna()

    if len(df_monthly) < 24:
        print("Not enough data. Skipping.")
        continue

    y = df_monthly['Quantity_Sold']
    X_full = df_monthly.drop(columns=['Quantity_Sold'])

    train_size = int(len(df_monthly) * 0.8)

    y_train = y[:train_size]
    y_test = y[train_size:]

    X_train_full = X_full[:train_size]
    X_test_full = X_full[train_size:]

    arima_model = ARIMA(y_train, order=(2,1,2))
    arima_result = arima_model.fit()

    arima_forecast = arima_result.forecast(steps=len(y_test))

    fitted_values = arima_result.fittedvalues
    fitted_values = fitted_values.reindex(y_train.index)

    residuals = y_train - fitted_values
    residuals = residuals.dropna()

    X_residual = X_train_full.loc[residuals.index]

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    xgb.fit(X_residual, residuals)

    residual_forecast = xgb.predict(X_test_full)

    final_forecast = arima_forecast.values + residual_forecast

    mae_arima = mean_absolute_error(y_test, arima_forecast)
    rmse_arima = np.sqrt(mean_squared_error(y_test, arima_forecast))

    mae_hybrid = mean_absolute_error(y_test, final_forecast)
    rmse_hybrid = np.sqrt(mean_squared_error(y_test, final_forecast))

    improvement = ((rmse_arima - rmse_hybrid) / rmse_arima) * 100

    print("ARIMA MAE :", round(mae_arima, 2))
    print("ARIMA RMSE:", round(rmse_arima, 2))
    print("Hybrid MAE :", round(mae_hybrid, 2))
    print("Hybrid RMSE:", round(rmse_hybrid, 2))
    print("RMSE Improvement:", round(improvement,2), "%")


    plt.figure(figsize=(12,6))

    plt.plot(y_train.index, y_train, label="Train")
    plt.plot(y_test.index, y_test, label="Actual", linewidth=2)
    plt.plot(y_test.index, arima_forecast, label="ARIMA Forecast")
    plt.plot(y_test.index, final_forecast, label="Hybrid Forecast", linewidth=2)

    plt.title(f"Hybrid ARIMA + XGBoost Forecast - {product_name}")
    plt.xlabel("Date")
    plt.ylabel("Quantity Sold")
    plt.legend()
    plt.grid()
    plt.show()