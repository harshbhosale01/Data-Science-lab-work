import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 

# Read the dataset
ds = pd.read_csv('Datasets/Mall_Customers.csv') 

# Print the head of the dataset
print(ds.head())

# Check for missing values
print(ds.isnull().sum())

# Plotting the scatterplots
sns.scatterplot(x=ds['Age'], y=ds['Annual Income (k$)'], data=ds)
sns.scatterplot(x=ds['Annual Income (k$)'], y=ds['Spending Score (1-100)'], data=ds) 


sns.pairplot(ds)

# Extracting features
x = ds.iloc[:, [3, 4]].values 

# Importing KMeans and suppressing warnings
import warnings
from sklearn.cluster import KMeans 

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42) 
    y_predict = kmeans.fit_predict(x) 

# Plotting the clusters
plt.scatter(x[y_predict==0, 0], x[y_predict==0, 1], s=50, c='blue', marker='+', label='Cluster 1') 
plt.scatter(x[y_predict==1, 0], x[y_predict==1, 1], s=50, c='green', marker='o', label='Cluster 2') 
plt.scatter(x[y_predict==2, 0], x[y_predict==2, 1], s=50, c='red', label='Cluster 3') 
plt.scatter(x[y_predict==3, 0], x[y_predict==3, 1], s=50, c='cyan', label='Cluster 4') 
plt.scatter(x[y_predict==4, 0], x[y_predict==4, 1], s=50, c='magenta', label='Cluster 5') # centroid 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='x', label='Centroid')
plt.title('Clusters of customers') 
plt.xlabel('Annual Income (k$)') 
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

