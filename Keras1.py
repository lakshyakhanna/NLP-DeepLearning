# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:10:16 2017

@author: lakshya.khanna
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,KFold
import pandas as pd
import os
import pydot
import graphviz
import numpy as np
from sklearn.metrics import accuracy_score

os.chdir('E:\POC\Hospital Readmission POC\Code\Keras')

X_train = pd.read_csv('X_train_var_imp.csv',nrows=2116)
X_test = pd.read_csv('X_test_var_imp.csv')
y_train = pd.read_csv('y_train.csv')
y_test= pd.read_csv('y_test.csv')

X_train.head()
y_train.shape
np.array(X_train).shape

y_train_dummy = to_categorical(y_train)

def base_model():
    model = Sequential()
    model.add(Dense(32,input_dim = 370,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile('adam','categorical_crossentropy',metrics=['accuracy'])
    return model

history1 = model.fit(np.array(X_train), np.array(y_train_dummy), verbose=0, epochs=20)
estimator = KerasClassifier(build_fn=base_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True, random_state=12)


results = cross_val_score(estimator, np.array(X_train), np.array(y_train_dummy), cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

estimator.fit(np.array(X_train),np.array(y_train_dummy))

y_pred = estimator.predict(np.array(X_test))

score = accuracy_score(y_pred,y_test)
