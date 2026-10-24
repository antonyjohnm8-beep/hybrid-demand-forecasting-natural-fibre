import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from xgboost import XGBRegressor


file_path = "coir_sales_dataset_2019_2025.csv"
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month


numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

df = df.dropna().drop_duplicates()

le = LabelEncoder()
df['Product_Encoded'] = le.fit_transform(df['Product'])


Q1 = df['Quantity_Sold'].quantile(0.25)
Q3 = df['Quantity_Sold'].quantile(0.75)
IQR = Q3 - Q1

df = df[
    (df['Quantity_Sold'] >= Q1 - 1.5 * IQR) &
    (df['Quantity_Sold'] <= Q3 + 1.5 * IQR)
]

plt.figure(figsize=(10, 6))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.show()

products = df['Product'].unique()

for product_name in products:

    print("\n" + "=" * 60)
    print(f"Processing Product: {product_name}")
    print("=" * 60)


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


    # Lag Features
    df_monthly['lag_1'] = df_monthly['Quantity_Sold'].shift(1)
    df_monthly['lag_3'] = df_monthly['Quantity_Sold'].shift(3)
    df_monthly['lag_6'] = df_monthly['Quantity_Sold'].shift(6)

    # Rolling Statistics
    df_monthly['rolling_mean_3'] = (
        df_monthly['Quantity_Sold'].rolling(3).mean()
    )

    df_monthly['rolling_std_3'] = (
        df_monthly['Quantity_Sold'].rolling(3).std()
    )

    # Seasonality Features
    df_monthly['month'] = df_monthly.index.month

    df_monthly['month_sin'] = np.sin(
        2 * np.pi * df_monthly.index.month / 12
    )

    df_monthly['month_cos'] = np.cos(
        2 * np.pi * df_monthly.index.month / 12
    )

    df_monthly = df_monthly.dropna()

    if len(df_monthly) < 24:
        print("Not enough data. Skipping.")
        continue


    y = df_monthly['Quantity_Sold']
    X = df_monthly.drop(columns=['Quantity_Sold'])

    train_size = int(len(df_monthly) * 0.8)

    y_train = y[:train_size]
    y_test = y[train_size:]

    X_train = X[:train_size]
    X_test = X[train_size:]

    result = adfuller(y_train)

    print("\nADF Statistic:", result[0])
    print("p-value:", result[1])


    arima = ARIMA(y_train, order=(2, 1, 2))

    arima_fit = arima.fit()

    arima_pred = arima_fit.forecast(len(y_test))


    sarima = SARIMAX(
        y_train,
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    sarima_fit = sarima.fit(disp=False)

    sarima_pred = sarima_fit.forecast(len(y_test))


    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    xgb.fit(X_train, y_train)

    xgb_pred = xgb.predict(X_test)


    residuals = y_train - arima_fit.fittedvalues
    residuals = residuals.dropna()

    X_res = X_train.loc[residuals.index]

    xgb_res = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    xgb_res.fit(X_res, residuals)

    res_pred = xgb_res.predict(X_test)

    hybrid_arima = arima_pred.values + res_pred


    sarima_residuals = y_train - sarima_fit.fittedvalues
    sarima_residuals = sarima_residuals.dropna()

    X_res_s = X_train.loc[sarima_residuals.index]

    xgb_res_s = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    xgb_res_s.fit(X_res_s, sarima_residuals)

    res_pred_s = xgb_res_s.predict(X_test)

    hybrid_sarima = sarima_pred.values + res_pred_s


    def evaluate(true, pred):

        mae = mean_absolute_error(true, pred)

        rmse = np.sqrt(mean_squared_error(true, pred))

        mape = mean_absolute_percentage_error(true, pred) * 100

        return (
            round(mae, 2),
            round(rmse, 2),
            round(mape, 2)
        )

    results = {
        "ARIMA": evaluate(y_test, arima_pred),
        "SARIMA": evaluate(y_test, sarima_pred),
        "XGBoost": evaluate(y_test, xgb_pred),
        "Hybrid ARIMA": evaluate(y_test, hybrid_arima),
        "Hybrid SARIMA": evaluate(y_test, hybrid_sarima)
    }

    print("\n--- Model Comparison ---")

    for model, (mae, rmse, mape) in results.items():

        print(
            f"{model}: "
            f"MAE={mae}, "
            f"RMSE={rmse}, "
            f"MAPE={mape}%"
        )

    plt.figure(figsize=(12, 6))

    plt.plot(
        y_train.index,
        y_train,
        label="Train"
    )

    plt.plot(
        y_test.index,
        y_test,
        label="Actual",
        linewidth=2
    )

    plt.plot(
        y_test.index,
        arima_pred,
        label="ARIMA"
    )

    plt.plot(
        y_test.index,
        sarima_pred,
        label="SARIMA"
    )

    plt.plot(
        y_test.index,
        xgb_pred,
        label="XGBoost",
        linestyle='dashed'
    )

    plt.plot(
        y_test.index,
        hybrid_arima,
        label="Hybrid ARIMA",
        linewidth=2
    )

    plt.plot(
        y_test.index,
        hybrid_sarima,
        label="Hybrid SARIMA",
        linewidth=2
    )

    plt.title(f"Model Comparison - {product_name}")

    plt.legend()

    plt.grid()

    plt.show()