
# IMPORTS


import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# FIX RANDOMNESS

random.seed(42)

np.random.seed(42)

# DATA


file_path = "coir_sales_dataset_2019_2025.csv"

df = pd.read_csv(file_path)

# DATE PROCESSING


df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(by='Date')

# HANDLE MISSING VALUES

numeric_cols = df.select_dtypes(include=np.number).columns

df[numeric_cols] = df[numeric_cols].fillna(
    df[numeric_cols].median()
)

df = df.drop_duplicates()


# OPTIONAL OUTLIER REMOVAL

Q1 = df['Quantity_Sold'].quantile(0.25)

Q3 = df['Quantity_Sold'].quantile(0.75)

IQR = Q3 - Q1

df = df[
    (df['Quantity_Sold'] >= Q1 - 1.5 * IQR) &
    (df['Quantity_Sold'] <= Q3 + 1.5 * IQR)
]


# CORRELATION MATRIX


plt.figure(figsize=(10, 6))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()


# STORE OVERALL RESULTS


overall_results = []


# EVALUATION FUNCTION


def evaluate(true, pred):

    mae = mean_absolute_error(true, pred)

    rmse = np.sqrt(mean_squared_error(true, pred))

    mape = mean_absolute_percentage_error(true, pred) * 100

    return (
        round(mae, 2),
        round(rmse, 2),
        round(mape, 2)
    )


# PRODUCT LOOP


products = df['Product'].unique()

for product_name in products:

    print("\n" + "=" * 60)

    print(f"Processing Product: {product_name}")

    print("=" * 60)


    # FILTER PRODUCT
   
    df_product = df[df['Product'] == product_name].copy()

    df_product = df_product.set_index('Date')

 
    # MONTHLY RESAMPLING
   

    df_monthly = df_product.resample('MS').agg({

        'Quantity_Sold': 'sum',
        'Price': 'mean',
        'Festival_Indicator': 'mean',
        'Rainfall_mm': 'mean',
        'Raw_Material_Cost_per_kg': 'mean',
        'Production_Volume': 'sum'

    })

    df_monthly = df_monthly.asfreq('MS')

    df_monthly.index.freq = 'MS'

   
    # FEATURE ENGINEERING
  
    # LAG FEATURES

    df_monthly['lag_1'] = (
        df_monthly['Quantity_Sold'].shift(1)
    )

    df_monthly['lag_3'] = (
        df_monthly['Quantity_Sold'].shift(3)
    )

    df_monthly['lag_6'] = (
        df_monthly['Quantity_Sold'].shift(6)
    )

    df_monthly['lag_12'] = (
        df_monthly['Quantity_Sold'].shift(12)
    )

    # ROLLING FEATURES

    df_monthly['rolling_mean_3'] = (
        df_monthly['Quantity_Sold'].rolling(3).mean()
    )

    df_monthly['rolling_mean_6'] = (
        df_monthly['Quantity_Sold'].rolling(6).mean()
    )

    df_monthly['rolling_std_3'] = (
        df_monthly['Quantity_Sold'].rolling(3).std()
    )

    df_monthly['rolling_std_6'] = (
        df_monthly['Quantity_Sold'].rolling(6).std()
    )

    # EMA FEATURES

    df_monthly['ema_3'] = (
        df_monthly['Quantity_Sold']
        .ewm(span=3)
        .mean()
    )

    df_monthly['ema_6'] = (
        df_monthly['Quantity_Sold']
        .ewm(span=6)
        .mean()
    )

    # DIFFERENCE FEATURES

    df_monthly['diff_1'] = (
        df_monthly['Quantity_Sold'].diff(1)
    )

    df_monthly['diff_12'] = (
        df_monthly['Quantity_Sold'].diff(12)
    )

    # DATE FEATURES

    df_monthly['month'] = (
        df_monthly.index.month
    )

    df_monthly['quarter'] = (
        df_monthly.index.quarter
    )

    df_monthly['trend'] = (
        np.arange(len(df_monthly))
    )

    # CYCLIC FEATURES

    df_monthly['month_sin'] = np.sin(
        2 * np.pi * df_monthly.index.month / 12
    )

    df_monthly['month_cos'] = np.cos(
        2 * np.pi * df_monthly.index.month / 12
    )

    # DROP NAN

    df_monthly = df_monthly.dropna()

   
    # DATA SIZE CHECK
    
    if len(df_monthly) < 24:

        print("Not enough data. Skipping.")

        continue

   
    # TARGET + FEATURES


    y = df_monthly['Quantity_Sold']

    X = df_monthly.drop(columns=['Quantity_Sold'])

    # TRAIN TEST SPLIT

    train_size = int(len(df_monthly) * 0.8)

    y_train = y[:train_size]

    y_test = y[train_size:]

    X_train = X[:train_size]

    X_test = X[train_size:]

 
    # ADF TEST

    result = adfuller(y_train)

    print("\nADF Statistic:", result[0])

    print("p-value:", result[1])


    # FEATURE SCALING

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

 
    # ARIMA MODEL


    arima = ARIMA(
        y_train,
        order=(1,1,1)
    )

    arima_fit = arima.fit()

    arima_pred = arima_fit.forecast(len(y_test))


    # SARIMA MODEL


    sarima = SARIMAX(

        y_train,

        order=(0,1,1),

        seasonal_order=(0,1,1,12),

        enforce_stationarity=False,

        enforce_invertibility=False

    )

    sarima_fit = sarima.fit(

        disp=False,

        maxiter=300,

        method='lbfgs'

    )

    sarima_pred = sarima_fit.forecast(len(y_test))

    # XGBOOST MODEL

    xgb = XGBRegressor(

        n_estimators=600,

        learning_rate=0.02,

        max_depth=5,

        subsample=0.95,

        colsample_bytree=0.95,

        objective='reg:squarederror',

        random_state=42,

        n_jobs=1,

        tree_method='exact'

    )

    xgb.fit(

        X_train_scaled,

        y_train

    )

    xgb_pred = xgb.predict(X_test_scaled)

    # HYBRID ARIMA-XGB


    arima_residuals = (
        y_train - arima_fit.fittedvalues
    )

    arima_residuals = arima_residuals.dropna()

    X_res_arima = X_train.loc[
        arima_residuals.index
    ]

    scaler_arima = StandardScaler()

    X_res_arima_scaled = scaler_arima.fit_transform(
        X_res_arima
    )

    X_test_arima_scaled = scaler_arima.transform(
        X_test
    )

    xgb_arima_res = XGBRegressor(

        n_estimators=400,

        learning_rate=0.02,

        max_depth=4,

        subsample=0.9,

        colsample_bytree=0.9,

        objective='reg:squarederror',

        random_state=42,

        n_jobs=1,

        tree_method='exact'

    )

    xgb_arima_res.fit(

        X_res_arima_scaled,

        arima_residuals

    )

    arima_residual_pred = xgb_arima_res.predict(
        X_test_arima_scaled
    )

    hybrid_arima_pred = (

        arima_pred.values +

        arima_residual_pred

    )

    # HYBRID SARIMA-XGB

    sarima_residuals = (
        y_train - sarima_fit.fittedvalues
    )

    sarima_residuals = sarima_residuals.dropna()

    X_res_sarima = X_train.loc[
        sarima_residuals.index
    ]

    scaler_sarima = StandardScaler()

    X_res_sarima_scaled = scaler_sarima.fit_transform(
        X_res_sarima
    )

    X_test_sarima_scaled = scaler_sarima.transform(
        X_test
    )

    xgb_sarima_res = XGBRegressor(

        n_estimators=800,

        learning_rate=0.015,

        max_depth=5,

        min_child_weight=1,

        subsample=0.95,

        colsample_bytree=0.95,

        gamma=0,

        reg_alpha=0.1,

        reg_lambda=1,

        objective='reg:squarederror',

        random_state=42,

        n_jobs=1,

        tree_method='exact'

    )

    xgb_sarima_res.fit(

        X_res_sarima_scaled,

        sarima_residuals

    )

    # RESIDUAL PREDICTION

    sarima_residual_pred = xgb_sarima_res.predict(
        X_test_sarima_scaled
    )

    # ADVANCED HYBRID

    hybrid_residual = (

        sarima_pred.values +

        sarima_residual_pred

    )

    hybrid_sarima_pred = (

        0.7 * hybrid_residual +

        0.3 * xgb_pred

    )

    # EVALUATION


    results = {

        "ARIMA": evaluate(y_test, arima_pred),

        "SARIMA": evaluate(y_test, sarima_pred),

        "XGBoost": evaluate(y_test, xgb_pred),

        "Hybrid ARIMA-XGB": evaluate(
            y_test,
            hybrid_arima_pred
        ),

        "Hybrid SARIMA-XGB": evaluate(
            y_test,
            hybrid_sarima_pred
        )

    }

    print("\n--- Model Comparison ---")

    for model, (mae, rmse, mape) in results.items():

        print(
            f"{model}: "
            f"MAE={mae}, "
            f"RMSE={rmse}, "
            f"MAPE={mape}%"
        )

        overall_results.append({

            'Product': product_name,

            'Model': model,

            'MAE': mae,

            'RMSE': rmse,

            'MAPE': mape

        })


    # FORECAST GRAPH

    plt.figure(figsize=(15, 7))

    plt.plot(
        y_train.index,
        y_train,
        label='Train'
    )

    plt.plot(
        y_test.index,
        y_test,
        label='Actual',
        linewidth=3
    )

    plt.plot(
        y_test.index,
        arima_pred,
        label='ARIMA'
    )

    plt.plot(
        y_test.index,
        sarima_pred,
        label='SARIMA'
    )

    plt.plot(
        y_test.index,
        xgb_pred,
        label='XGBoost'
    )

    plt.plot(
        y_test.index,
        hybrid_arima_pred,
        label='Hybrid ARIMA-XGB',
        linewidth=3
    )

    plt.plot(
        y_test.index,
        hybrid_sarima_pred,
        label='Hybrid SARIMA-XGB',
        linewidth=4
    )

    plt.title(
        f"Forecast Comparison - {product_name}"
    )

    plt.xlabel("Date")

    plt.ylabel("Quantity Sold")

    plt.legend()

    plt.grid()

    plt.show()

    # BAR GRAPH COMPARISON


    models = list(results.keys())

    mae_values = [results[m][0] for m in models]

    rmse_values = [results[m][1] for m in models]

    mape_values = [results[m][2] for m in models]

    x = np.arange(len(models))

    width = 0.25

    plt.figure(figsize=(14, 7))

    plt.bar(
        x - width,
        mae_values,
        width,
        label='MAE'
    )

    plt.bar(
        x,
        rmse_values,
        width,
        label='RMSE'
    )

    plt.bar(
        x + width,
        mape_values,
        width,
        label='MAPE'
    )

    plt.xticks(x, models, rotation=10)

    plt.ylabel("Error Value")

    plt.title(
        f"Model Performance Comparison - {product_name}"
    )

    plt.legend()

    plt.grid(axis='y')

    plt.show()

# ============================================================
# OVERALL PERFORMANCE ANALYSIS
# ============================================================

overall_df = pd.DataFrame(overall_results)

avg_results = overall_df.groupby('Model')[[
    'MAE',
    'RMSE',
    'MAPE'
]].mean().reset_index()

print("\n")
print("=" * 70)
print("OVERALL AVERAGE PERFORMANCE")
print("=" * 70)

print(avg_results)

# BEST MODEL

best_model = avg_results.loc[
    avg_results['MAPE'].idxmin()
]

print("\n")
print("=" * 70)

print(
    f"BEST OVERALL MODEL: "
    f"{best_model['Model']}"
)

print(
    f"Average MAPE: "
    f"{round(best_model['MAPE'], 2)}%"
)

print("=" * 70)

# OVERALL COMPARISON GRAPH

models = avg_results['Model']

mae_avg = avg_results['MAE']

rmse_avg = avg_results['RMSE']

mape_avg = avg_results['MAPE']

x = np.arange(len(models))

width = 0.25

plt.figure(figsize=(15, 7))

plt.bar(
    x - width,
    mae_avg,
    width,
    label='Average MAE'
)

plt.bar(
    x,
    rmse_avg,
    width,
    label='Average RMSE'
)

plt.bar(
    x + width,
    mape_avg,
    width,
    label='Average MAPE'
)

plt.xticks(x, models, rotation=10)

plt.ylabel("Average Error")

plt.title(
    "Overall Model Performance Comparison"
)

plt.legend()

plt.grid(axis='y')

plt.show()

# FINAL MAPE GRAPH

plt.figure(figsize=(12, 6))

bars = plt.bar(
    avg_results['Model'],
    avg_results['MAPE']
)

plt.ylabel("Average MAPE (%)")

plt.title(
    "Overall Forecast Accuracy Comparison"
)

plt.grid(axis='y')

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f'{height:.2f}%',
        ha='center',
        va='bottom'
    )

plt.show()