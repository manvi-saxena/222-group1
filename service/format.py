import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn 
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from numpy import ndarray
from sklearn.decomposition import PCA as pca
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import time

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

def mypca(filename1, number):
    number = int(number) 
    x,y=data_format(filename1)
    pcafif=pca(n_components=number, whiten=False, random_state=7)
    pcafif.fit(x)
    result = pcafif.explained_variance_ratio_
    list = result.tolist()
    json_str = json.dumps(list)   
    return(json_str)
    
def normalize(data):
    data= (data-np.min(data))/(np.max(x)-np.min(data)).values
    return data

# return ('first 15 components',  "\n", pcafif.explained_variance_ratio_, '\n\nPCAs\n',"15 components", sum(pcafif.explained_variance_ratio_), '\n')
    return json_str
