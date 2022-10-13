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

# import csv files
Maths = pd.read_csv("maths.csv")
Portuguese = pd.read_csv("Portuguese.csv")

# settings options to view the entire data set in terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

###############   Mean values

# mean values for all entries based on 'school'
print('Mean values according to school:')
print(Maths.groupby('school').mean())
print()

# mean values for all entries based on 'sex'
print('Mean values according to sex:')
print(Maths.groupby('sex').mean())
print()

#