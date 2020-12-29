#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 05:27:29 2020

@author: tylerpruitt
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Take out data and assign names to columns of data
labels = ['sepal-length', 'sepal-width', 'petal-length', 'petal_width', 'class']
dataset = pd.read_csv('iris.csv',names=labels)

# Print information about data to screen
print(dataset.describe())
print('')
print(dataset.groupby('class').size())
print('')

# Print out plots of data
dataset.plot(kind = 'box', subplots = True, layout = (2,2), sharex = False, sharey = False)
dataset.hist()
pd.plotting.scatter_matrix(dataset)

# Split up data into arrays
array = dataset.values #np.array of entire data set
X = array[:,0:4] #np.array of numerical values
Y = array[:,4] #np.array of names
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=0.20,random_state=1)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print('')
# Check our accuracy for the name validation data
print(accuracy_score(Y_validation, predictions))
print('')
# Check confusion matrix for 3 binary classes: Iris-setosa, Iris-versicolor, Iris-virginica
print(confusion_matrix(Y_validation, predictions))
print('')
# Check the general classification report
print(classification_report(Y_validation, predictions))