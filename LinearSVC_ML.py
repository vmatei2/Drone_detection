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

def plot_classification_report(cr, title='Classification Report', with_avg_total = False, cmap = plt.cm.Blues):
    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2 : len(lines) - 5]:
        t = line.split()
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)
    if with_avg_total :
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)

    plt.imshow(plotMat, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    x_ticks_marks = np.arange(3)
    y_ticks_marks = np.arange(len(classes))
    plt.xticks(x_ticks_marks, ['precision', 'recall', 'f1-score'], rotation = 45)
    plt.yticks(y_ticks_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.title(title)
    plt.show()

def initial_data_visualization(data):
    print("The size of the data file is: ",data.shape , "\n")
    print(data.groupby('Signal_type').size(), "\n")
    print("The first 5 rows from the dataset : \n", data.head())
    print("The last 5 rows from the dataset : \n", data.tail())

def create_scatter_plot(feature1, feature2, data):
    scatterplot = sns.scatterplot(x = feature1, y = feature2 , hue = 'Signal_type' , data = data)
    feature_name = feature1.split('_')[1]
    if (feature_name == "std"):
        plt.title('Scatter plot for standard deviation')
    elif (feature_name == "mean"):
        plt.title('Scatter plot for mean')
    elif (feature_name == "mad"):
        plt.title("Scatter plot for mean absolute deviation")
    elif (feature_name == "kurtosis"):
        plt.title("Scatter plot for kurtosis")
    plt.grid()
    plt.show()

def create_correlation_heat_map(data):
    heatmap = sns.heatmap(data.corr(), annot = True)
    plt.title("Correlation heat map of statistical features of IQ signals")
    plt.show()

def split_data(data, test_size, random_state):
    array = data.values
    X = array[:,0:8] #features
    Y = array[:,8] #drone/bt/wifi
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = random_state)
    return X_train, X_test, y_train, y_test

def implement_algorithm(algorithm, X_train, y_train, X_test, y_test, algorithm_name):
    algorithm.fit(X_train, y_train)
    y_pred = algorithm.predict(X_test)
    print("\n The", algorithm_name, " has an accuracy of :", "{:.2f}".format(metrics.accuracy_score(y_test, y_pred)))
    print("Confussion matrix for the", algorithm_name , " implemented")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    cm_df = pd.DataFrame(cm, index = ['Drone', 'Bluetooth', 'WiFi'],
                      columns = ['Drone', 'Bluetooth', 'WiFi'])
    plt.figure(figsize = (5.5,4))
    sns.heatmap(cm_df, annot = True)
    plt.title(algorithm_name + " Accuracy:{0:.3f}".format(metrics.accuracy_score(y_test, y_pred)))
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    print("\n Classification report following the tests: ")
    cr = classification_report(y_test, y_pred)
    print(cr)
    plot_classification_report(cr, algorithm_name)
    print("\n\n")

def main():
    #loading data from URL
    url = 'https://raw.githubusercontent.com/vmatei2/Drone_detection/master/Features.csv'
    names = ['I_std','I_mean','I_mad','Q_std', 'Q_mean'	,'Q_mad','I_kurtosis','Q_kurtosis','Signal_type']
    data = pd.read_csv(url, names = names)

    #data visualization to better understand what we're working with
    initial_data_visualization(data)
    #create figure and axis
    create_scatter_plot("I_std", "Q_std" , data = data)
    create_scatter_plot("I_mean", "Q_mean", data = data)
    create_scatter_plot("I_mad", "Q_mad", data = data)
    create_scatter_plot("I_kurtosis", "Q_kurtosis", data = data)
    create_correlation_heat_map(data)
    X_train, X_test, y_train, y_test = split_data(data, test_size = 0.3, random_state = 59)


    clf = svm.SVC(kernel = 'linear') #linear Kernel
    implement_algorithm(clf, X_train, y_train, X_test, y_test, algorithm_name = "Linear SVC algorithm")

    #KneighborsClassifierModel
    neigh = KNeighborsClassifier(n_neighbors = 3)
    implement_algorithm(neigh, X_train, y_train, X_test, y_test, algorithm_name = "K Nearest Neighbors Classifier algorithm")

    #Creating GNB object
    gnb_model = GaussianNB()
    implement_algorithm(gnb_model, X_train, y_train, X_test, y_test, "Gaussian Naive Bayes algorithm")
    #testing on a new set of values
    X_new = [[27.2525, 0.875454,1.26262, 151.262, 1.9895, 326.21, 45.96 , 85.21 ]]
    ynew = gnb_model.predict(X_new)
    print('ynew is ', ynew)


main()