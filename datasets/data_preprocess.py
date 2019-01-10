import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Extracting data from dataset
data = pd.read_csv("Data.csv")

row = data.iloc[:,:-1].values
col = data.iloc[:,3].values

from sklearn.preprocessing import Imputer, LabelEncoder,OneHotEncoder
# Eliminating the missing values
impu = Imputer()
impu = Imputer(missing_values="NaN",strategy="mean",axis=0)
row[:, 1:3] = impu.fit_transform(row[:, 1:3])
#Encoding categorical data
label = LabelEncoder()
row[:,0] = label.fit_transform(row[:,0])
hot = OneHotEncoder()
hot = OneHotEncoder(categorical_features=[0])
row = hot.fit_transform(row).toarray()
label_X = LabelEncoder()
col = label_X.fit_transform(col)
 #splitting the dataset into train and test data.
from sklearn.cross_validation import train_test_split
i_train,i_test,d_train,d_test = train_test_split(row,col,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

i_train = sc_X.fit_transform(row)
i_test = sc_X.transform(row)



