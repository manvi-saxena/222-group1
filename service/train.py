import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as pca
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

path = os.path.dirname(__file__)

def data_format(filename):
    data=pd.read_csv(path + '/data/' + filename)
    data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
    data.diagnosis=[1 if element== "M" else 0 for element in data.diagnosis]
    x=data.drop('diagnosis', axis=1)
    y=data.diagnosis
    x = (x-np.min(x))/(np.max(x)-np.min(x)).values
  #  x = x.to_json() 
   # y = y.to_json()
    return (x, y)

def split_data(filename):
    x, y = data_format(filename)
    # x and y already exist as a variable (?)
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    return [x_train, x_test, y_train, y_test]

#def six_comp():
#    x, y = data_format():
#    x6 = x.iloc[:,0:6]
#    six = split_data(x6, y)
#    return six

def get_subset(filename, n): 
    x, y = data_format(filename)
    r = x.iloc[:, 0:n]
    return(r, y)

#given train data in function with 6 features
def logistic_regression(filename, n):
    n = int(n)
    x, y = get_subset(filename, n)
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    my_model = LogisticRegression()
    my_model.fit(x_train,y_train)
    y_pred = my_model.predict(x_train)
    score = my_model.score(x_train, y_train)
    return (score)

def logistic_regression_test(filename, n):
    n = int(n)
    x, y = get_subset(filename, n)
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    my_model = LogisticRegression()
    my_model.fit(x_train,y_train)
    y_pred = my_model.predict(x_test)
    score = my_model.score(x_test, y_test)
    return (score)


#requires user input for # of features
def logistic_regression_variable(n):
    n = abs(int(n))
    #raise error for invalid input?
    x = x.iloc[:, 0:n]
    my_model = LogisticRegression()
    my_model.fit(x_train,y_train)
    return my_model
