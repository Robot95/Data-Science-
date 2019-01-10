# Data Presprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Extracting data from dataset
data = pd.read_csv("50_Startups.csv")

X = data.iloc[:,:-1].values
Y = data.iloc[:,4].values

# Data Encoding

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

Encoder = LabelEncoder()
X[:,3] = Encoder.fit_transform(X[:,3])

hot_encoder = OneHotEncoder(categorical_features = [3])
X = hot_encoder.fit_transform(X).toarray() 

#Splitting data set into test set and train set

from sklearn.cross_validation import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#performing linear regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,Y_train)

y_pred = regressor.predict(X_test)

# To introduce a constant to the independent variable
import statsmodels.formula.api as sm 

X = np.append(arr = np.ones((50,1)).astype(int),values= X ,axis=1)
