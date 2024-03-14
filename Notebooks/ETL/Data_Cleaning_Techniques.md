
# Data Cleaning Techniques with Pandas and NumPy

## Basic Data Cleaning Operations

### Handling Missing Values

Pandas offers several methods to deal with missing data, such as removal and imputation.

```python
import pandas as pd
import numpy as np

# Example dataset
data = {
    'A': [1, np.nan, 3, 4, 5],
    'B': [6, 7, 8, np.nan, 10],
    'C': [11, 12, np.nan, np.nan, 15]
}
df = pd.DataFrame(data)

# Remove rows with missing values
df.dropna()

# Fill missing values with a specific value
df.fillna(value=0)

# Use forward fill to propagate the last valid observation forward
df.fillna(method='ffill')

# Use NumPy to compute mean for imputation
mean_value = df['A'].mean()
df['A'].fillna(value=mean_value, inplace=True)
```

### Removing Duplicates

Duplicate data can be easily identified and removed using Pandas.

```python
# Example dataset with duplicate rows
data = {
    'A': [1, 1, 2, 3, 4, 4],
    'B': ['a', 'a', 'b', 'c', 'd', 'd']
}
df = pd.DataFrame(data)

# Identifying duplicate rows
duplicates = df.duplicated()

# Removing duplicate rows
df.drop_duplicates(inplace=True)
```

## Advanced Data Cleaning Techniques

### Data Transformation with Pandas

Data transformation includes operations like normalization, conversion, and encoding categorical data.

```python
# Normalizing data (Min-Max scaling)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df['A'] = scaler.fit_transform(df[['A']])

# Converting data types
df['A'] = df['A'].astype('float64')

# Encoding categorical variables
df_encoded = pd.get_dummies(df, columns=['B'])
```

This notebook offers a starting point for data cleaning with Pandas and NumPy, covering handling missing values, removing duplicates, and basic data transformation techniques.
