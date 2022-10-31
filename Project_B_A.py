# Seth Ball
# Keller Sedillo
# Shafiq Zaman
# CS488  
# Final Project Basic Analysis
# 10/13/22

import os
import math
import pandas as pd 
import numpy as np
import statistics as stat
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# import csv files
Maths = pd.read_csv("maths.csv")
Maths_plot = Maths
Portuguese = pd.read_csv("Portuguese.csv")
Portuguese_Plot = Portuguese
#Portuguese = pd.read_csv('Portuguese_Fake_Data.csv')
'''
# append the fake data to the real data
Real.iloc[1: , :]
frames = [Real,Fake]
Portuguese = pd.concat(frames)
print(Portuguese)
'''
# settings options to view the entire data set in terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


###############   Mean values

# mean values for all entries int Maths based on 'school'
print('Mean values according to school:')
print('Math:')
print(Maths.groupby('school').mean(numeric_only=True))
print()
print('Portuguese:')
print(Portuguese.groupby('school').mean(numeric_only=True))
print()

# mean values for all entries in Maths based on 'sex'
print('Mean values according to sex:')
print('Math:')
print(Maths.groupby('sex').mean(numeric_only=True))
print()
print('Portuguese:')
print(Portuguese.groupby('sex').mean(numeric_only=True))
print()

# mean values for all entries in Maths based on 'absences'
print('Mean values according to absences')
print('Math:')
print(Maths.groupby('absences').mean(numeric_only=True))
print()
print('Portuguese:')
print(Portuguese.groupby('absences').mean(numeric_only=True))
print()

# mean values for all entries in Maths based on 'failures'
print('Mean values according to failures')
print('Math:')
print(Maths.groupby('failures').mean(numeric_only=True))
print()
print('Portuguese:')
print(Portuguese.groupby('failures').mean(numeric_only=True))
print()

# mean values for all entries in Maths based on 'study time'
print('Mean values according to study time')
print('Math:')
print(Maths.groupby('studytime').mean(numeric_only=True))
print()
print('Portuguese:')
print(Portuguese.groupby('studytime').mean(numeric_only=True))
print()

# mean values for all entries in Maths based on 'internet'
print('Mean values according to internet')
print('Math:')
print(Maths.groupby('internet').mean(numeric_only=True))
print()
print('Portuguese:')
print(Portuguese.groupby('internet').mean(numeric_only=True))
print()

# mean values for all entries in Maths based on 'Medu'
print('Mean values according to Mothers education')
print('Math:')
print(Maths.groupby('Medu').mean(numeric_only=True))
print()
print('Portuguese:')
print(Portuguese.groupby('Medu').mean(numeric_only=True))
print()

# mean values for all entries in Maths based on 'Fedu'
print('Mean values according to fathers education')
print('Math:')
print(Maths.groupby('Fedu').mean(numeric_only=True))
print()
print('Portuguese:')
print(Portuguese.groupby('Fedu').mean(numeric_only=True))
print()

# plots to show correlation between differnt term scores and the final grades
Maths_plot.plot(x = 'G1', y = 'G3', kind = 'scatter', label = 'G1 scores compared to G3 scores')
plt.show()

Maths_plot.plot(x = 'G1', y = 'G2', kind = 'scatter', label = 'G1 scores compared to G2 scores')
plt.show()

Maths_plot.plot(x = 'G2', y = 'G3', kind = 'scatter', label = 'G2 scores compared to G3 scores')
plt.show()


############################# Classification Section

# create training and testing data
# this will be commented out after the test and 
# training data is produced
y = Maths.iloc[-1]
cols = [-1]
X = Maths.drop(Maths.columns[cols], axis=1, inplace=True)

X, y = make_classification()
# Test and train models for Maths
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=1, test_size=0.2, shuffle=True)


E_scores = []
M_scores = []
K_vals = [1,3,5,7,9,11]

# Euclidean Distance KNN
for i in range(12):
	if i % 2 == 1:
		N = KNeighborsClassifier(n_neighbors=i)
		N.fit(x_train, y_train)

		score = N.score(x_test, y_test)
		#print(score)

		E_scores.append(score)

# Manhattan Distance KNN
for i in range(12):
	if i % 2 == 1:
		N = KNeighborsClassifier(n_neighbors=i, metric='manhattan')
		N.fit(x_train, y_train)

		score = N.score(x_test, y_test)
		#print(score)

		M_scores.append(score)

# plot for knn Euclidean Distance
plt.plot(E_scores, K_vals)
plt.title('KNN-Euclidean Distance')
plt.xlabel('Euclidean Scores')
plt.ylabel('K-Values')
plt.show()

# plot for knn Manhattan Distance
plt.plot(M_scores, K_vals)
plt.title('KNN-Manhattan Distance')
plt.xlabel('Manhattan Scores')
plt.ylabel('K-Values')
plt.show()
