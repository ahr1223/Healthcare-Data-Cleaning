import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Creating DataFrame
data = {
    "PatientID": range(1, 21),
    "Age": [44, 39, 49, 58, 35, 25, 46, 28, 60, 55, 41, 48, 58, 35, 67, 70, 43, 74, 19, 56],
    "BloodPressure": [118, 109, 149, 121, 109, 129, 132, 93, 145, 125, 143, 141, 93, 145, 176, 109, 148, 122, 147, 119],
    "SugarLevel": [87.89, 177.32, 144.14, 90.35, 126.42, 95.27, 146.60, 109.75, 103.19, 197.72, 
                   180.57, 181.97, 181.78, 133.38, 87.00, 193.27, 135.93, 129.41, 125.48, 160.71],
    "Weight": [105.57, 105.70, 77.78, 115.24, 70.38, 119.05, 62.17, 81.79, 94.63, 118.59, 
               103.58, 61.45, 50.68, 113.18, 84.93, 77.71, 106.57, 83.30, 74.08, 111.86]
}

df = pd.DataFrame(data)

# 1. Check for Missing Values and Fill if Necessary
df.fillna(df.mean(), inplace=True)  # Filling with mean (if there are missing values)

# 2. Remove Duplicates
df.drop_duplicates(inplace=True)

# 3. Handle Outliers using Z-score
z_scores = np.abs(stats.zscore(df[['BloodPressure', 'SugarLevel', 'Weight']]))  # Compute Z-score
df_cleaned = df[(z_scores < 3).all(axis=1)]  # Keep values with Z-score < 3 (removes extreme outliers)

# 4. Standardizing Data
scaler = StandardScaler()
df_cleaned[['BloodPressure', 'SugarLevel', 'Weight']] = scaler.fit_transform(df_cleaned[['BloodPressure', 'SugarLevel', 'Weight']])

# Display cleaned dataset
print("\nCleaned Healthcare Data:\n", df_cleaned)
