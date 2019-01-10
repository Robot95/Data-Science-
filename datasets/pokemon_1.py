import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

raw_data = pd.read_csv("Salary_Data.csv")

years = raw_data.iloc[:,-1].values
sal = raw_data.iloc[:,1].values


from sklearn.cross_validation import train_test_split

