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

# import csv files
Maths = pd.read_csv("maths.csv")
Portuguese = pd.read_csv("Portuguese.csv")

# settings options to view the entire data set in terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

###############   Mean values

# mean values for all entries int Maths based on 'school'
print('Mean values according to school:')
print('Math:')
print(Maths.groupby('school').mean())
print()
print('Portuguese:')
print(Portuguese.groupby('school').mean())
print()

# mean values for all entries in Maths based on 'sex'
print('Mean values according to sex:')
print('Math:')
print(Maths.groupby('sex').mean())
print()
print('Portuguese:')
print(Portuguese.groupby('sex').mean())
print()

# mean values for all entries in Maths based on 'absences'
print('Mean values according to absences')
print('Math:')
print(Maths.groupby('absences').mean())
print()
print('Portuguese:')
print(Portuguese.groupby('absences').mean())
print()

# mean values for all entries in Maths based on 'failures'
print('Mean values according to failures')
print('Math:')
print(Maths.groupby('failures').mean())
print()
print('Portuguese:')
print(Portuguese.groupby('failures').mean())
print()

# mean values for all entries in Maths based on 'study time'
print('Mean values according to study time')
print('Math:')
print(Maths.groupby('studytime').mean())
print()
print('Portuguese:')
print(Portuguese.groupby('studytime').mean())
print()

# mean values for all entries in Maths based on 'internet'
print('Mean values according to internet')
print('Math:')
print(Maths.groupby('internet').mean())
print()
print('Portuguese:')
print(Portuguese.groupby('internet').mean())
print()

# mean values for all entries in Maths based on 'Medu'
print('Mean values according to Mothers education')
print('Math:')
print(Maths.groupby('Medu').mean())
print()
print('Portuguese:')
print(Portuguese.groupby('Medu').mean())
print()

# mean values for all entries in Maths based on 'Fedu'
print('Mean values according to fathers education')
print('Math:')
print(Maths.groupby('Fedu').mean())
print()
print('Portuguese:')
print(Portuguese.groupby('Fedu').mean())
print()

# plots to show correlation between differnt term scores and the final grades
Maths.plot(x = 'G1', y = 'G3', kind = 'scatter', label = 'G1 scores compared to G3 scores')
plt.show()

Maths.plot(x = 'G1', y = 'G2', kind = 'scatter', label = 'G1 scores compared to G2 scores')
plt.show()

Maths.plot(x = 'G2', y = 'G3', kind = 'scatter', label = 'G2 scores compared to G3 scores')
plt.show()