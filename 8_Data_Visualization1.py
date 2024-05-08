# Importing necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import load_dataset

# Load the Titanic dataset
titanic_data = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(titanic_data.head())

# Visualize the data to find patterns
# Pairplot to visualize relationships between numeric variables
sns.pairplot(titanic_data, hue='survived')
plt.title('Pairplot of Titanic Dataset')
plt.show()

# Plot histogram of ticket prices
plt.figure(figsize=(10, 6))
sns.histplot(titanic_data['fare'], bins=30, kde=True)
plt.title('Histogram of Ticket Prices')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()
