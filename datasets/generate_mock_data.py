import pandas as pd
from datetime import timedelta, datetime
import numpy as np

# Function to generate random dates
def generate_dates(start_date, end_date, n, format='%Y-%m-%d'):
    date_range = pd.date_range(start_date, end_date)
    dates = np.random.choice(date_range, size=n, replace=True)

    # Convert dates to strings using strftime (outside NumPy array)
    formatted_dates = [pd.to_datetime(date).strftime(format) for date in dates]
    return formatted_dates

# Parameters for data generation
n = 20  # Number of records
start_date = '2023-01-01'
end_date = '2023-12-01'

np.random.seed(42)  # For reproducibility

# Generating data
data = {
    'date': generate_dates(start_date, end_date, n, format='%m/%d/%Y'),
    'impressions': np.random.randint(100, 10000, size=n),
    'clicks': np.random.randint(10, 1000, size=n),
    'spent': np.round(np.random.uniform(10, 500, size=n), 2)
}

# Creating DataFrame
df = pd.DataFrame(data)
df.sort_values(by='date', inplace=True)
df.reset_index(drop=True, inplace=True)
print(df)

# # Save to CSV
csv_file_path = '/media/nemsys/data/projects/courses/common/JupyterNotebooksExamples/Notebooks/ETL/datasets/facebook_ads.csv'
df.to_csv(csv_file_path, index=False)
