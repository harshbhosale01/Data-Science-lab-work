import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
ds = pd.read_csv('dslab_4.csv')
ds.head()
ds.isnull().sum()
sns.pairplot(ds)
plt.show()
x = ds.iloc[:,[1,2]].values

from sklearn.cluster import AgglomerativeClustering
agg_clustering = AgglomerativeClustering(n_clusters = 3 , affinity = 'euclidean',linkage ='ward')
y_pred = agg_clustering.fit_predict(x)
y_pred

plt.scatter(x[y_pred == 0,0],x[y_pred == 0,1],s=50,c='blue',marker='+',label='Cluster 1')
plt.scatter(x[y_pred == 1,0],x[y_pred == 1,1],s=50,c='green',marker='o',label='Cluster 2')
plt.scatter(x[y_pred == 2,0],x[y_pred == 2,1],s=50,c='red',label='Cluster 3')

plt.title('hierarchical clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

