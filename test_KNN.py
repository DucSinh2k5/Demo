
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "X": [1,2,3,6,7,8,3,6],
    "Y" : [2,3,3,5,7,6,7,2],
    "Label": ["A","A","A","B","B","B","A","B"]
}

df = pd.DataFrame(data).to_csv("Phan_Loai_Diem.csv", index= False)
dataset = pd.read_csv("Phan_Loai_Diem.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.2,random_state=1)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn = knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
print(Y_pred)
print(Y_test)
print("Accuracy: ", accuracy_score(Y_pred, Y_test))
print("Matrix:" , confusion_matrix(Y_pred,Y_test))
