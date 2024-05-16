import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r"E:\ProdigyInfoTech_DataScience_Tasks\train.csv" 
titanic_df = pd.read_csv(file_path)


print("First few rows of the Titanic dataset:")
print(titanic_df.head())


print("\nMissing Values:")
print(titanic_df.isnull().sum())

# Data Cleaning
median_age = titanic_df['Age'].median()
titanic_df['Age'].fillna(median_age, inplace=True)

# Exploratory Data Analysis (EDA)
# Summary statistics
print("\nSummary Statistics:")
print(titanic_df.describe())


plt.figure(figsize=(10, 6))
sns.histplot(titanic_df['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


gender_survival = titanic_df.groupby('Sex')['Survived'].mean()
print("\nSurvival Rates by Gender:")
print(gender_survival)


class_survival = titanic_df.groupby('Pclass')['Survived'].mean()
print("\nSurvival Rates by Passenger Class:")
print(class_survival)

# Visualize Relationships
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)
plt.title('Survival Rates by Passenger Class and Gender')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()