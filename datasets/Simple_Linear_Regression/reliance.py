import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("call_list.csv")

recharge = data.iloc[:,:1].values
amount = data.iloc[:,1].values

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(recharge,amount,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
data_calls = LinearRegression()
data_calls.fit(recharge,amount)

prd = data_calls.predict(x_test)

plt.scatter(x_train,y_train)
plt.plot(x_train,data_calls.predict(x_train))
plt.title("best amount to recharge with")
plt.xlabel("No of days of recharge")
plt.ylabel("Amount to pay")
plt.show()

plt.scatter(x_test,y_test)
plt.plot(x_train,data_calls.predict(x_train))
plt.title("best amount to recharge with")
plt.xlabel("No of days of recharge")
plt.ylabel("Amount to pay")
plt.show()


