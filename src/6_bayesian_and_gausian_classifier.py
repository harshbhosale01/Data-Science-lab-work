import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
ds=pd.read_csv('SALARY_DATA.csv')
ds.head()
x=ds.iloc[:,[2,3]].values 
y=ds.iloc[:,4].values 
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)
from sklearn.linear_model import LogisticRegression
classifier =LogisticRegression (random_state=42)
classifier.fit(xtrain, ytrain)
y_pred= classifier.predict(xtest)
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(ytest,y_pred)
print("Confusion Matrix: \n",cm)
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(ytest, y_pred))
from matplotlib.colors import ListedColormap
x_set,y_set = xtest, ytest
#creating a mesh grid start, stop , step for both Age and Estimate salary
x1, x2 =np.meshgrid(np.arange(start= x_set[:,0].min()-1,stop=x_set[:,0].max()+1, step=0.01), np.arange(start = x_set[:,1].min()-1,stop=x_set[:,1].max()+1, step=0.01))
#contour plot
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
  plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],c=ListedColormap(('red','green')) (i), label=j)
plt.title('Classifier(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(xtrain,ytrain)
ypred2 =gnb.predict(xtest)
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(ytest,y_pred2)
print("Confusion Matrix: \n",cm)
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(ytest, y_pred2))
from matplotlib.colors import ListedColormap
x_set,y_set = xtest, ytest
#creating a mesh grid start, stop , step for both Age and Estimate salary
x1, x2 =np.meshgrid(np.arange(start= x_set[:,0].min()-1,stop=x_set[:,0].max()+1, step=0.01), np.arange(start = x_set[:,1].min()-1,stop=x_set[:,1].max()+1, step=0.01))
#contour plot
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
  plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],c=ListedColormap(('red','green')) (i), label=j)
plt.title('Classifier(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()