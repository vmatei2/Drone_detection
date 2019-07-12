# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:07:16 2019

@author: vladm
"""

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report

url = 'https://raw.githubusercontent.com/vmatei2/Drone_detection/master/Features.csv'
filename = 'Features.csv'
names = ['I_Std','I_mean','I_mad','Q_std', 'Q_mean'	,'Q_mad','I_kurtosis','Q_kurtosis','Signal_type']
data = pd.read_csv(url, names = names)
print("The size of the data file is:", data.shape)
print()
print(data.groupby('Signal_type').size()) #prints how many positive/negative values for drones we have
print()
print("The first 5 rows from the dataset:")
print(data.head())
print("The last 5 rows from the data set:")
print(data.tail())
array = data.values
X = array[:,0:8]  #features
Y = array[:,8]  #drone/Bt/Wifi
test_size = 0.3 #70% training, 30% testing
random_state = 59
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state =random_state)
clf = svm.SVC(kernel = 'linear') #linear Kernel
clf.fit(X_train, y_train) #train the model 
#predict the response
y_pred = clf.predict(X_test)
print()
print("The Linear SVC algorithm's accuracy is:",metrics.accuracy_score(y_test, y_pred))
print()
print("Confusion Matrix for the Linear SVC Algorithm implemented")
print(confusion_matrix(y_test,y_pred))
print()
print("Classification report following the tests: ")
print(classification_report(y_test,y_pred))
print()
print()
neigh = KNeighborsClassifier(n_neighbors = 3)
#training the model 
neigh.fit(X_train, y_train)
#predict the response
pred = neigh.predict(X_test)
#accuracy of K neighbors
print("Kneighbors accuracy score:",metrics.accuracy_score(y_test,pred))
print("Confusion Matrix for the KNeighborsClassifier Algorithm implemented")
print(confusion_matrix(y_test,pred))
print()
print("Classification report following the tests: ")
print(classification_report(y_test,y_pred))
print()
#Creating GNB object
gnb_model = GaussianNB()
#train the model 
gnb_model.fit(X_train,y_train)
pred_gnb = gnb_model.predict(X_test)
#accuracy of GNB
print("The Naive-Bayes accuracy score:",metrics.accuracy_score(y_test,pred_gnb))
print("Confusion Matrix for the Naive-Bayes Algorithm implemented")
print(confusion_matrix(y_test,pred_gnb))
print()
print("Classifcation report following the tests: ")
print(classification_report(y_test,pred_gnb))
X_new = [[27.2525, 0.875454,1.26262, 151.262, 1.9895, 326.21, 45.96 , 85.21 ]]

ynew = gnb_model.predict(X_new)
print('ynew is ',ynew)
