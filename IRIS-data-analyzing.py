# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:16:26 2016

@author: Sachi Angle
"""

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

iris_data = pd.read_csv('iris_data.csv', na_values = ['NA'])
sb.pairplot(iris_data.dropna(),hue='class' )

plt.figure(figsize=(10, 10))

for column_index, column in enumerate(iris_data.columns):
    if column == 'class':
        continue
    plt.subplot(2, 2, column_index + 1)
    sb.violinplot(x='class', y=column, data=iris_data)