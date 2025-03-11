# Healthcare Data Cleaning and Preprocessing

## Overview
This project demonstrates the process of cleaning and preprocessing a healthcare dataset. It includes handling missing values, removing duplicates, detecting and removing outliers, and standardizing numerical features to prepare the data for further analysis or machine learning.

## Dataset Description
The dataset consists of 20 patient records with the following attributes:
- **PatientID**: Unique identifier for each patient
- **Age**: Patient's age
- **BloodPressure**: Patient's blood pressure measurement
- **SugarLevel**: Blood sugar level in mg/dL
- **Weight**: Patient's weight in kg

## Features and Processing Steps
1. **Handling Missing Values**:
   - If any data is missing, it is replaced with the column mean.

2. **Removing Duplicates**:
   - Duplicate records are removed to maintain data integrity.

3. **Outlier Detection and Removal**:
   - Outliers are identified using the **Z-score method**, where values with Z-scores greater than 3 standard deviations are removed.

4. **Data Standardization**:
   - The `StandardScaler` from Scikit-learn is used to normalize BloodPressure, SugarLevel, and Weight so that they have a mean of 0 and a standard deviation of 1.

## Requirements
To run this project, you need to install the following Python libraries:
```sh
pip install pandas numpy scipy scikit-learn
```

## Code Implementation
```python
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
```

## Expected Output
The cleaned dataset is printed, showing normalized values for BloodPressure, SugarLevel, and Weight, with missing values replaced and outliers removed.

## Conclusion
This project demonstrates fundamental data preprocessing techniques that are essential for ensuring high-quality data. The cleaned dataset is now ready for further analysis or machine learning applications.

## References
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)

