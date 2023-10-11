import numpy as np
import pandas as pd
import pylab as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

datatxt = np.loadtxt('olympic100m.txt', delimiter=',')
data = pd.DataFrame(datatxt,columns=['Year','Time'])
pd.plotting.scatter_matrix(data, alpha=0.2,diagonal='hist', hist_kwds={'bins':20})

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

X_in = np.array([[5, 2.9, 1, 0.2]])
print(X_in.shape)
y_pred = knn.predict(X_in)
print(f'This is classifed as: {y_pred}')
y_pred_label = iris_dataset['target_names'][y_pred]
print(iris_dataset['target_names'])

y_pred = knn.predict(X_test)
print(iris_dataset['target_names'][y_pred])
#print(np.mean(y_pred == y_test))
print(knn.score(X_test, y_test))