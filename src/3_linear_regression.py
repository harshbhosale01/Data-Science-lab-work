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
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)
y_pred=lin_reg.predict(x_test)
import matplotlib.pyplot as plt
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred,color='blue')
plt.title('salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')

