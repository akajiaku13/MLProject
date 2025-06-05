import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('notebook/data/stud.csv')
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check for duplicates
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Check data types
print("\nData types of each column:")
print(df.dtypes)

# Check unique values in categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
print("\nUnique values in categorical columns:")
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}")

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Obtain numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("\nNumeric columns:")
print(numeric_cols)
print('\nCategorical columns:')
print(categorical_cols)

# Adding columns for 'Total Score' and 'Average Score'
df['Total Score'] = df[numeric_cols].sum(axis=1)
df['Average Score'] = df[numeric_cols].mean(axis=1)

# Visualization of distributions
# Histogram and KDE
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(1, 2, 1)
sns.histplot(df['Average Score'], kde=True, bins=30, color='green')
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='Average Score', kde=True, hue='gender')
plt.show()

# Export new DataFrame with additional columns
# df.to_csv('notebook/data/stud_with_scores.csv', index=False)