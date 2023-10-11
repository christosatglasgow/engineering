import numpy as np
import pandas as pd
import pylab as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error 

datatxt = np.loadtxt('olympic100m.txt', delimiter=',')
data = pd.DataFrame(datatxt,columns=['Year','Time'])
pd.plotting.scatter_matrix(data, alpha=0.2,diagonal='hist', hist_kwds={'bins':20})

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

X_in = np.array([[5, 2.9, 1, 0.2]])
#print(X_in.shape)
y_pred = knn.predict(X_in)
#print(f'This is classifed as: {y_pred}')
y_pred_label = iris_dataset['target_names'][y_pred]
#print(iris_dataset['target_names'])

y_pred = knn.predict(X_test)
#print(iris_dataset['target_names'][y_pred])
#print(np.mean(y_pred == y_test))
#print(knn.score(X_test, y_test))

training_accuracy = []
testing_accuracy = []
neighbors = range(1,51)

for n_neighbors in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    testing_accuracy.append(knn.score(X_test, y_test))

plt.figure()
plt.plot(neighbors, training_accuracy,label='Training accuracy')
plt.plot(neighbors, testing_accuracy,label='Testing accuracy')
plt.xlabel('Neighbors')
plt.ylabel('Accuracy')
plt.legend()

#knn regression with olypmic data
X_train, X_test, y_train, y_test = train_test_split(
    data.Year, data.Time, random_state=0)
reg = KNeighborsRegressor(n_neighbors=2)
X_train = np.array([X_train]).T
X_test = np.array([X_test]).T
                   
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f'R^2 is: {reg.score(X_test,y_test)}')
mse = mean_squared_error(y_test,y_pred)
print(f'This is the MSE: {mse}')
