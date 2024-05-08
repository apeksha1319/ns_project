# Step 1: Importing required libraries
import pandas as pd
import numpy as np

# Step 3: Load the dataset into pandas DataFrame
# Assuming the dataset file is named "titanic.csv" and located in the current directory
data = pd.read_csv("train.csv")

# Step 4: Data Preprocessing

# Check for missing values in the data
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Get initial statistics using describe()
initial_stats = data.describe()
print("\nInitial Statistics:\n", initial_stats)

# Check dimensions of the DataFrame
dimensions = data.shape
print("\nDimensions of DataFrame:", dimensions)

# Step 5: Data Formatting and Normalization
# Summarize types of variables by checking data types
data_types = data.dtypes
print("\nTypes of Variables:\n", data_types)

# If variables are not in the correct data type, apply proper type conversions.
# For example, if 'Age' column is a float but should be an integer:
# data['Age'] = data['Age'].astype(int)

# Turn categorical variables into quantitative variables in Python
# For example, if 'Sex' column is categorical (male/female), we can encode it as 0 and 1.
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
