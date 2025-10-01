import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score

df = pd.read_csv("eda_data.csv")
x = df.drop(columns=['Salary Estimate', 'min_salary', 'max_salary', 'avg_salary'])
y=df['avg_salary']

category_cols = x.select_dtypes(include=['object']).columns

x_encoded = pd.get_dummies(x,columns=category_cols, drop_first=True)
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y,test_size=0.3,random_state=42)

regressor = LinearRegression()
regressor.fit(x_train, y_train)
mse = cross_val_score(regressor,x_train,y_train,scoring='neg_mean_squared_error',cv=5)

mean_mse = np.mean(mse)

y_pred = regressor.predict(x_test)
r2_score(y_test, y_pred)

print(mean_mse)
print(r2_score(y_test,y_pred))

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE on test set: {mae:.2f}")
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()
