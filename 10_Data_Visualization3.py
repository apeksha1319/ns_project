# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv(url, names=column_names)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris_data.head())

# Information about features and their types
print("\nFeatures and their types:")
print(iris_data.dtypes)

# Create histograms for each feature
iris_data.hist(figsize=(10, 8))
plt.suptitle('Histograms of Features in Iris Dataset')
plt.show()

# Create box plots for each feature
plt.figure(figsize=(10, 8))
sns.boxplot(data=iris_data)
plt.title('Boxplots of Features in Iris Dataset')
plt.show()

# Identify outliers
# Let's consider outliers as values that are outside 1.5 times the interquartile range (IQR) from the first and third quartiles
Q1 = iris_data.quantile(0.25)
Q3 = iris_data.quantile(0.75)
IQR = Q3 - Q1
outliers = ((iris_data < (Q1 - 1.5 * IQR)) | (iris_data > (Q3 + 1.5 * IQR))).any(axis=1)
print("\nNumber of outliers:")
print(outliers.sum())
print("\nOutliers:")
print(iris_data[outliers])
