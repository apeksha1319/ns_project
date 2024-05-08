import pandas as pd

# first part

# Load the dataset
data = pd.read_csv("data.csv")

# Group by a categorical variable and calculate summary statistics for numeric variables
summary_stats = data.groupby('categorical_variable').agg({'numeric_variable': ['mean', 'median', 'min', 'max', 'std']})

# Rename columns for clarity
summary_stats.columns = ['Mean', 'Median', 'Minimum', 'Maximum', 'Standard Deviation']

# Print summary statistics
print(summary_stats)

# Create a list containing a numeric value for each response to the categorical variable
unique_categories = data['categorical_variable'].unique()
category_numeric_values = list(range(1, len(unique_categories)+1))
category_numeric_mapping = dict(zip(unique_categories, category_numeric_values))
print("Numeric mapping for categorical variable:")
print(category_numeric_mapping)


# second part

import pandas as pd

# Load the dataset
iris_data = pd.read_csv("iris.csv")

# Filter rows for species 'Iris-versicolor' and 'Iris-virginica'
versicolor_data = iris_data[iris_data['Species'] == 'Iris-versicolor']
virginica_data = iris_data[iris_data['Species'] == 'Iris-virginica']

# Display basic statistical details for each species
print("Statistical Details for Iris-versicolor:")
print(versicolor_data.describe())

print("\nStatistical Details for Iris-virginica:")
print(virginica_data.describe())
