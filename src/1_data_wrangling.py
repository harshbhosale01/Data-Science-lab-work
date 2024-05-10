import pandas as pd
ds = pd.read_csv('SALARY DATA.csv')
ds.head()
ds.tail()
ds.dtypes
duplicate_rows = ds[ds.duplicated()]
# Print duplicate rows
print("Duplicate Rows:")
print(duplicate_rows)
#To remove the duplicate data
ds.drop_duplicates(inplace=True)
missing_rows = ds[ds.isnull().any(axis=1)]
print(missing_rows)
# Remove rows with missing values
ds_cleaned = ds.dropna()
