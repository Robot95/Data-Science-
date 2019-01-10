import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Ex
raw_data = pd.read_csv("Salary_Data.csv")

years = raw_data.iloc[:,:1].values
sal = raw_data.iloc[:,1].values


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(years,sal,test_size=1/3,random_state=0)

#Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

sal_pred = regressor.predict(x_test)
#visualising the training set results
plt.scatter(x_train,y_train,color = "red")
plt.plot(x_train,regressor.predict(x_train), color="blue")
plt.title("salary and experience' one training set'")
plt.xlabel("Experience of employee")
plt.ylabel("Salary od employess")
plt.show()

#visualising the test set results
plt.scatter(x_test,y_test,color = "red")
plt.plot(x_train,regressor.predict(x_train), color="blue")
plt.title("salary and experience' one training set'")
plt.xlabel("Experience of employee")
plt.ylabel("Salary od employess")
plt.show()



