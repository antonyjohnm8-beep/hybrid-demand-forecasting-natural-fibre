import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

file_path = "coir_sales_dataset_2019_2025.csv"   
df = pd.read_csv(file_path)

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values(by='Date')

df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter

print("\nMissing Values:")
print(df.isnull().sum())

#Fill numeric missing values with median
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


# plt.figure(figsize=(10,6))
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.show()

numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()



plt.figure(figsize=(12,6))
for product in df['Product'].unique():
    temp = df[df['Product'] == product]
    plt.plot(temp['Date'], temp['Quantity_Sold'], label=product)

plt.title("Monthly Sales Trend by Product")
plt.xlabel("Date")
plt.ylabel("Quantity Sold")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Rainfall_mm'], y=df['Quantity_Sold'])
plt.title("Rainfall vs Sales")
plt.show()
