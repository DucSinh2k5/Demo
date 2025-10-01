
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv("bmi.csv")
X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values

ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#KNN with k = 15
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
#lr
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred2 = lr.predict(X_test)

#KNN with k = 105

knn2 = KNeighborsClassifier(n_neighbors=105)
knn2.fit(X_train, Y_train)
Y_pred2 = knn2.predict(X_test)

print("Accuracy of KNN with k = 15: " , accuracy_score(Y_test,Y_pred ))
print("Accuracy of KNN with k = 105: " , accuracy_score(Y_test,Y_pred2 ))
print("Accuracy of Logistic Regression: ",  accuracy_score(Y_test,Y_pred2 ))



