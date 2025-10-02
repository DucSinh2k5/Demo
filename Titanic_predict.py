
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection  import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv("tested.csv")

X = df[["Pclass", "Sex", "Age"]]
Y = df[["Survived"]]

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X["Age"] = imputer.fit_transform(X[["Age"]])

ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder(), ["Sex"])], remainder="passthrough")
X = ct.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)



print("Accuracy: ", accuracy_score(Y_pred, Y_test))

