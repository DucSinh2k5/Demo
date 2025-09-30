
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]
df.to_csv("Iris_datasets.csv", index=False)



dataset = pd.read_csv("Iris_datasets.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=1)

knn = KNeighborsClassifier(n_neighbors=5)
knn = knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
print(Y_pred)
print(Y_test)

print("Accuracy: ", accuracy_score(Y_pred, Y_test))









