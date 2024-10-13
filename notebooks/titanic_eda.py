import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('./data/train.csv')

# Data Overview
print(df.head())

# Missing Values
sns.heatmap(df.isnull(), cbar=False)
plt.show()

# Survival rate by gender
sns.barplot(x='Sex', y='Survived', data=df)
plt.show()

# Class distribution of survivors
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.show()

# Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()