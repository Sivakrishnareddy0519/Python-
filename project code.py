import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sivakrishna=pd.read_csv("C:\\Users\\sivak\\Downloads\\Mental_Health_Care_in_the_Last_4_Weeks.csv")
print(sivakrishna)
siva=sivakrishna.head(50)
print(siva)
sivakrishna.isnull().sum()
sivakrishna.dropna(inplace=True)
print("Duplicate rows:", sivakrishna.duplicated().sum())
print(sivakrishna.describe())
print(sivakrishna.info())
sns.scatterplot(x="HighCI",y="LowCI",data=siva)
plt.title("Scatter plot of LowCI and HighCI")
plt.show()
numeric = siva.select_dtypes(include=[np.number])
corr=numeric.corr()
sns.heatmap(corr, annot=True,cmap="RdBu",fmt=".2f")
plt.title("Correlation of Time Period,Value,LowCI,HighCI")
plt.show()
sns.barplot(x="HighCI",y="State",data=siva)
plt.title("Barplot of HighCI")
plt.show()
sns.barplot(x="LowCI",y="State",data=siva,color='green')
plt.title("Barplot of LowCI")
plt.show()
sns.boxplot(x="LowCI",data=siva)
plt.title("Boxplot of LowCI")
plt.show()
sns.boxplot(x="HighCI",data=siva)
plt.title("Boxplot of HighCI")
plt.show()
sns.pairplot(data=siva)
plt.show()
sns.histplot(data=siva,bins=30)
plt.title("Hisplot of siva")
plt.show()
sns.countplot(x="LowCI",data=siva)
plt.title("Count plot of LowCI")
plt.show()
sns.countplot(x="HighCI",data=siva)
plt.title("Count plot of HighCI")
plt.show()

# Z-Test using fresh data (not affected by dropna)
original_data = pd.read_csv("C:\\Users\\sivak\\Downloads\\Mental_Health_Care_in_the_Last_4_Weeks.csv")

# Filter for Male and Female with non-null CI values
male_data = original_data[(original_data['Subgroup'] == 'Male') & original_data['LowCI'].notnull() & original_data['HighCI'].notnull()]
female_data = original_data[(original_data['Subgroup'] == 'Female') & original_data['LowCI'].notnull() & original_data['HighCI'].notnull()]

# Make sure we have data
if male_data.empty or female_data.empty:
    print("Not enough data to perform Z-test. Please check the dataset.")
else:
    mean_male = male_data['Value'].mean()
    mean_female = female_data['Value'].mean()

    std_male = ((male_data['HighCI'] - male_data['LowCI']) / (2 * 1.96)).mean()
    std_female = ((female_data['HighCI'] - female_data['LowCI']) / (2 * 1.96)).mean()

    n_male = male_data.shape[0]
    n_female = female_data.shape[0]

    if std_male == 0 or std_female == 0:
        print("Standard deviation is zero. Cannot perform Z-test.")
    else:
        z = (mean_female - mean_male) / np.sqrt((std_female**2 / n_female) + (std_male**2 / n_male))
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z)))

        print("\nZ-Test Results (Original Data):")
        print(f"Mean (Female): {mean_female:.2f}")
        print(f"Mean (Male): {mean_male:.2f}")
        print(f"Z-score: {z:.2f}")
        print(f"P-value: {p_value:.4f}")

        if p_value < 0.05:
            print("Conclusion: Statistically significant difference between male and female prescription usage.")
        else:
            print("Conclusion: No statistically significant difference.")
