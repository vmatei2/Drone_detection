# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:07:24 2019

@author: vladm
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf


features = pd.read_csv('Features_tf.csv') #iris
print(features.head())
print(features.dtypes)
#changing from 64 bits to 32 for use with tensorflow
features.iloc[:,0:8] = features.iloc[:,0:8].astype(np.float32) 
print(features.dtypes)
#converting signal types to integers
features['Signal_type'] = features['Signal_type'].map({'WIFI':0,'BT':1,'Drone':2})

#test size, random_state
test_size = 0.3
rand = 45


#splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(features.iloc[:,0:8], features["Signal_type"], test_size = test_size, random_state = rand)

print(X_train.shape)
print(X_test.shape)

#constructing the DNN classifier
#DNN classifier expects following inputs: feature_columns-map data to the model
#hidden_units : no of hidden units in each layer. All layers would be fully connected
#n_classes: No. of classes
columns = features.columns[0:8]

#all the features in the dataset are real valued and continous

feature_columns = [tf.contrib.layers.real_valued_column(k) for k in columns]

#making the input function
#estimators in tensorflow.contrib.learn accept inputfunctions that feed the data to the model during training
# they return a mappiong of the feature columns to the tensors that contain the data for the features columns along with labels
#here I create a dictionary where the keys are the feature columns which map to the tensors containing values for these features

def input_fn(df,labels):
    feature_cols = {k:tf.constant(df[k].values, shape = [df[k].size,1])for k in columns}
    label = tf.constant(labels.values, shape = [labels.size,1])
    return feature_cols, label

#construcintg the DNN classifier
classifier = tf.contrib.learn.DNNClassifier(feature_columns=  feature_columns, hidden_units = [10,20,10],n_classes = 3,model_dir = "/tmp/Drone_model")
classifier.fit(input_fn = lambda: input_fn(X_train, y_train), steps = 100)

#Evaluating the DNN classifier
ev = classifier.evaluate(input_fn = lambda: input_fn(X_test, y_test),steps = 100)['accuracy']
print('\nAccuracy: {0:f}'.format(ev))






