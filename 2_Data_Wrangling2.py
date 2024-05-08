import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load the dataset
data = pd.read_csv("xAPI-Edu-Data.csv")

# 1. Handling missing values and inconsistencies
# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)

# Handle missing values by imputation or removal
# For numerical variables, impute with mean or median
data['numerical_var'] = data['numerical_var'].fillna(data['numerical_var'].median())
# For categorical variables, impute with mode or remove the rows
data['categorical_var'] = data['categorical_var'].fillna(data['categorical_var'].mode()[0])

# Handle inconsistencies in categorical variables
# For example, remove rows where category is not in predefined list

# 2. Handling outliers in numeric variables
# Visualize boxplot for numeric variables
numeric_vars = ['numeric_var1', 'numeric_var2']
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[numeric_vars])
plt.title('Boxplot of Numeric Variables')
plt.show()

# Define function to identify outliers using z-score
def detect_outliers_z_score(data):
    outliers = []
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    for i in data:
        z_score = (i - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers

# Identify and handle outliers using z-score
for var in numeric_vars:
    outliers = detect_outliers_z_score(data[var])
    data = data[~data[var].isin(outliers)]

# 3. Apply data transformations
# Log transformation on a numeric variable to reduce skewness
data['log_transformed_var'] = np.log(data['numeric_var1'])

# Visualize the distribution before and after transformation
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, )
