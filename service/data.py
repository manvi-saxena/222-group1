#This is the data downloading module and it also includes plotting defs 

import requests
import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from os import listdir
from flask import Flask, request, send_file, make_response

#Obviously we should use a text file and a post to get this value and read it in here.

# This implementation allows the user to place a file in called input.txt in the dir called input
# The structure of this could be imporved

code_dir = os.path.dirname(__file__)
url = 'https://drive.google.com/uc?export=download&id=1s7Krt_X5id39718dwEcJMRx4Vk47N0g4 '
def get_url():
    input_path = code_dir+'/input/input.txt'
    input_file = open(input_path, "rt")
    contents = input_file.read()
    url = contents.rstrip()
    input_file.close()
    return str(url)

def new_download(filename):
    url = get_url()
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)

def download_data(url, filename):
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)
    return 
    
def download(output):
    data_dir = code_dir+'/data/'
    output_file = data_dir+output
    new_download(filename=output_file)
    return  str(output) + " Downloaded" + " to" + str(code_dir)


def generate_figure(filename):
    data_dir = code_dir+'/data/'
    file = data_dir + filename
    with open(file,'r') as csvfile:
        my_file = pd.read_csv(csvfile)
        nfl = my_file
        nfl_numeric = nfl.select_dtypes(include=[np.number])
        nfl_numeric.boxplot()
        bytes_image = io.BytesIO()
        bytes_image
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
    return bytes_image

def generate_figureNorm(filename):
    data_dir = code_dir+'/data/'
    file_path = data_dir + filename
    with open(file_path,'r') as csvfile:
        my_file = pd.read_csv(csvfile)
        nfl = my_file
        nfl_numeric = nfl.select_dtypes(include=[np.number])
        nfl_normalized = (nfl_numeric - nfl_numeric.mean()) / nfl_numeric.std()
        nfl_normalized.boxplot()
        bytes_image = io.BytesIO()
        bytes_image
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
    return bytes_image

def create_boxplot(filename):
    bytes_obj = generate_figure(filename)
    
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

def create_boxplotNorm(filename):
    bytes_obj = generate_figureNorm(filename)
    
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')


def data_partition(filename1, ratio):
    data_dir = code_dir+'/data/'
    file_path = data_dir + filename1
    file = open(filename1,'r')
    training_file=data_dir+'train_'+str(ratio)+'.csv'
    test_file=data_dir+'test_'+ str(ratio)+'.csv'
    data = file.readlines()
    count = 0
    size = len(data)
    ftrain =open(training_file,'w')
    ftest =open(test_file,'w')
    for line in data:
        if(count< int(size*ratio)):
            ftrain.write(line)
        else:
            ftest.write(line)
        count = count + 1  

def partition(filename,ratio):
    ratio = float(ratio)
    data_dir = code_dir+'/data/'
    path=data_dir+filename
    data_partition(path,ratio)
    return "Successfully Partitioned"

def numeric_data(filename):
    data_dir = code_dir+'/data/'
    file_path = data_dir+filename
    print(os.getcwd())
    with open(file_path, 'r') as csvfile:
         file = pd.read_csv(csvfile)
         file_num = file.select_dtypes(include=[np.number])
         file_num = file_num.astype(float)
         file_num.to_csv(code_dir+'/data/'+'num_'+filename)
         return "Removed non numeric comlumns"

def loadCSV(fileName):
    fileHandler = open(fileName, "rt")
    lines = fileHandler.readlines()
    fileHandler.close()
    del lines[0] # remove the header
    dataset = []
    for line in lines:
        instance = lineToTuple(line)
        instance = np.delete(instance,0,0)
        dataset.append(instance)
    return array(dataset,dtype=np.float32)

def lineToTuple(line):
    cleanLine = line.strip()                  # remove leading/trailing whitespace and newlines
    cleanLine = cleanLine.replace('"', '')# get rid of quotes
    lineList = cleanLine.split(",")          # separate the fields
    stringsToNumbers(lineList)            # convert strings into numbers
    lineTuple = array(lineList)
    return lineTuple

def stringsToNumbers(myList):
    for i in range(len(myList)):
        if (isValidNumberString(myList[i])):
         myList[i] = float(myList[i])
    return 

def isValidNumberString(s):
  if len(s) == 0:
    return False
  if len(s) > 1 and s[0] == "-":
    s = s[1:]
  for c in s:
    if c not in "0123456789.":
      return False
  return True
