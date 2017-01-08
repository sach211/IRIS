# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 19:49:33 2016

@author: Sachi Angle"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

iris_data = pd.read_csv('iris_data.csv', na_values = ['NA'])
all_inputs = iris_data[['s_length','s_width','p_length','p_width']].values
all_classes = iris_data['class'].values

from sklearn.cross_validation import train_test_split

(training_inputs, testing_inputs, training_classes, testing_classes) = train_test_split(all_inputs, all_classes, train_size=0.75, random_state=1)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(training_inputs, training_classes)
sc = dtc.score(testing_inputs, testing_classes)

model_accuracies = []
for i in range(1000):
    (training_inputs, testing_inputs, training_classes, testing_classes) = train_test_split(all_inputs, all_classes, train_size=0.75)
    dtc.fit(training_inputs, training_classes)
    sc = dtc.score(testing_inputs, testing_classes)
    model_accuracies.append(sc)

sb.distplot(model_accuracies)

from sklearn.cross_validation import cross_val_score

cv_scores = cross_val_score(dtc, all_inputs, all_classes, cv = 10)
sb.distplot(cv_scores)
plt.title('Average score: {}'.format(np.mean(cv_scores))) 

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

p = {'max_depth':[1,2,3,4,5],'max_features':[1,2,3,4]}
c_v = StratifiedKFold(all_classes, n_folds = 10)

grid_search = GridSearchCV(dtc, param_grid = p, cv = c_v)
grid_search.fit(all_inputs,all_classes)
dtc = grid_search.best_estimator_
print(grid_search.best_params_)
print(grid_search.best_score_)
