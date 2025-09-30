
# from sklearn.datasets import fetch_california_housing
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import accuracy_score, mean_squared_error

# housing = fetch_california_housing(as_frame=True)
# df = housing.frame
# df.to_csv("california_housing.csv", index=False)


# datasets = pd.read_csv("california_housing.csv")

# X = datasets.iloc[:, :-1].values
# Y = datasets.iloc[:,-1].values

# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=42)

# regressor = LinearRegression()
# regressor = regressor.fit(X_train, Y_train)

# # print(regressor.coef_)
# # print(regressor.intercept_)

# Y_pred = regressor.predict(X_test)
# print(Y_pred[0])
# print(Y_test[0])

# print("Baseline MSE:", mean_squared_error(Y_test, Y_pred))


# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# pipe = Pipeline([
#     ("scaler", StandardScaler()),
#     ("linreg", LinearRegression())
# ])

# pipe.fit(X_train, Y_train)
# y_pred = pipe.predict(X_test)

# print("Scaled Linear Regression MSE:", mean_squared_error(Y_test, y_pred))


# from sklearn.preprocessing import PolynomialFeatures

# pipe_poly = Pipeline([
#     ("scaler", StandardScaler()),
#     ("poly", PolynomialFeatures(degree=2, include_bias=False)),
#     ("linreg", LinearRegression())
# ])

# pipe_poly.fit(X_train, Y_train)
# y_pred = pipe_poly.predict(X_test)

# print("Polynomial Features MSE:", mean_squared_error(Y_test, y_pred))


# from sklearn.ensemble import RandomForestRegressor

# rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
# rf.fit(X_train, Y_train)
# y_pred = rf.predict(X_test)

# print("Random Forest MSE:", mean_squared_error(Y_test, y_pred))



import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


housing = fetch_california_housing(as_frame=True)
df = housing.frame
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Scaled + Linear": Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression())
    ]),
    "Polynomial (deg=2)": Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("linreg", LinearRegression())
    ]),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
}

plt.figure(figsize=(15,10))

for i, (name, model) in enumerate(models.items(), 1):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    plt.subplot(2, 3, i)
    plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') 
    plt.title(f"{name}\nMSE={mse:.3f}")
    plt.xlabel("Giá trị thật")
    plt.ylabel("Giá trị dự đoán")

plt.tight_layout()
plt.show()
