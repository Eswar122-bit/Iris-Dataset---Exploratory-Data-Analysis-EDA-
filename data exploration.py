# Project 2: Data Exploration on Iris Dataset

# 1. Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 2. Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Iris dataset loaded successfully!\n")
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# 3. Explore structure
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nClass Distribution:\n", df['species'].value_counts())

# 4. Check for missing values
print("\nMissing Values Check:")
print(df.isnull().sum())

# 5. Visualizations

# Histogram of each feature
df.hist(figsize=(10, 8))
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Pairplot
sns.pairplot(df, hue="species", diag_kind="kde")
plt.show()

# Boxplot for sepal length
plt.figure(figsize=(10, 6))
sns.boxplot(x="species", y="sepal length (cm)", data=df)
plt.title("Sepal Length Distribution by Species")
plt.show()

# Heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns='species').corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
