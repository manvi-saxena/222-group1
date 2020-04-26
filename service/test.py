import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn 
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as pca
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import io
from flask import Flask, request, send_file, make_response

path = os.path.dirname(__file__)

def log_reg(x):
    x = x.iloc[:, 0:6]
    split_data()
    my_model = LogisticRegression()
    my_model.fit(x_test, y_test)
    return my_model

def log_reg_input(x, number):
    number = abs(int(number))
    x = x.iloc[:, 0:number]
    split_data()
    my_model = LogisticRegression()
    my_model.fit(x_test, y_test)
    return my_model

# from scikit learn source

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    bytes_image = io.BytesIO()
    bytes_image
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image

# Important things to note are io.BytesIO() and send_file magicial powers, io from the io module and send_file is flask

def gen_cof_mat(filename, n, arg1,arg2):
    n = int(n)
    x, y = get_subset(filename, n)
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    c_value = float(arg1)
    my_model = LogisticRegression(C = c_value, solver = arg2)
    my_model.fit(x_train,y_train)
    y_pred = my_model.predict(x_test)
    np.set_printoptions(precision=2) #it may be needed
    # Plot non-normalized confusion matrix
    # Plot normalized confusion matrix
    class_names = ['mag', 'non'] # make these names better spell them out
    class_names = np.asarray(class_names)
    #class_names = class_names.reshape((1,-1))
    bytes_obj = plot_confusion_matrix(y_test, y_pred, classes = class_names)

    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')


def logistic_regression_test(filename, n):
    n = int(n)
    x, y = get_subset(filename, n)
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    my_model = LogisticRegression()
    my_model.fit(x_train,y_train)
    y_pred = my_model.predict(x_test)
    score = my_model.score(x_test, y_test)
    return (score)

def get_subset(filename, n): 
    x, y = data_format(filename)
    r = x.iloc[:, 0:n]
    return(r, y)

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
