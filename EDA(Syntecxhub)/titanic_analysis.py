import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load
df = pd.read_csv("titanic.csv")

print("Dataset loaded successfully")
print(df.head())
print("-" * 50)

# info
print("Dataset information:")
print(df.info())
print("-" * 50)

print("Missing values before cleaning:")
print(df.isnull().sum())
print("-" * 50)

# clean
df["Age"].fillna(df["Age"].median(), inplace=True)
df.drop("Cabin", axis=1, inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
print("Missing values after cleaning:")
print(df.isnull().sum())
print("-" * 50)

# visualization
sns.set(style="whitegrid")

sns.countplot(x="Survived", data=df)
plt.title("Survival Count (0 = Not Survived, 1 = Survived)")
plt.savefig("survival_count.png")
plt.close()

# survival bu gender
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.savefig("survival_by_gender.png")
plt.close()

# survival by passenger
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.savefig("survival_by_class.png")
plt.close()

# age
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.savefig("age_distribution.png")
plt.close()

print("All charts created and saved successfully!")
