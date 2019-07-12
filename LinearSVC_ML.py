# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:07:16 2019

@author: vladm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report

#Plot decision function for a 2D SVC

def plot_svc(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    #create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y,X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P= model.deicion_function(xy).reshape(X.shape)
    #plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0 ,1], alpha = 0.5,
               linestyle=['--','-','--'])
    
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

#loading data from URL
url = 'https://raw.githubusercontent.com/vmatei2/Drone_detection/master/Features.csv'
names = ['I_std','I_mean','I_mad','Q_std', 'Q_mean'	,'Q_mad','I_kurtosis','Q_kurtosis','Signal_type']
data = pd.read_csv(url, names = names)

#data visualization to better understand what we're working with
print("The size of the data file is:", data.shape)
print()
print(data.groupby('Signal_type').size()) #prints how many positive/negative values for drones we have
print()
print("The first 5 rows from the dataset:")
print(data.head())
print("The last 5 rows from the data set:")
print(data.tail())
#create figure and axis
sns.scatterplot(x='I_std', y='Q_std',hue = 'Signal_type', data = data)
plt.title('Scatter plot for Standard deviation')
plt.show()

sns.scatterplot(x='I_mean', y='Q_mean',hue = 'Signal_type', data = data)
plt.title('Scatter plot for mean')
plt.show()

sns.scatterplot(x='I_mad', y='Q_mad',hue = 'Signal_type', data = data)
plt.title('Scatter plot for mean absolute deviation')
plt.show()

sns.scatterplot(x='I_kurtosis', y='Q_kurtosis',hue = 'Signal_type', data = data)
plt.title('Scatter plot for kurtosis')
plt.show()

sns.heatmap(data.corr(), annot=True)
plt.show()

array = data.values
X = array[:,0:8]  #features
Y = array[:,8]  #drone/Bt/Wifi

test_size = 0.3 #70% training, 30% testing
random_state = 59

#splitting test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state =random_state)

#Linear Svc Model
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




#KneighborsClassifierModel
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

#testing on a new set of values
X_new = [[27.2525, 0.875454,1.26262, 151.262, 1.9895, 326.21, 45.96 , 85.21 ]]
ynew = gnb_model.predict(X_new)
print('ynew is ',ynew)
