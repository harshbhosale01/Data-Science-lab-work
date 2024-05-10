import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
ds = pd.read_csv('salary.csv')
ds.head()
ds.tail()
ds.dtypes
x = ds.iloc[:,:-1].values ; y = ds.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import Ridge, Lasso
rd = Ridge(alpha=3)
rd.fit(x_train,y_train)
rd.score(x_test,y_test)
y_pred1=rd.predict(x_test)
import matplotlib.pyplot as plt
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred,color='blue')
plt.title('salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')

ls = Lasso(alpha=3)
ls.fit(x_train,y_train)
ls.score(x_test,y_test)
y_pred2=ls.predict(x_test)
import matplotlib.pyplot as plt
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred2,color='blue')
plt.title('salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')