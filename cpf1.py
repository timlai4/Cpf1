# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 21:19:37 2019

@author: Tim
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import interp
#from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

train_cpf1_df = pd.read_csv("cpf1-train.txt", sep='\t')
test_cpf1_df = pd.read_csv("cpf1-test.txt",sep='\t')
new_columns = ['50bp', '34bp', '20bp', 'Background Indel Frequency', 
               'Background Indel read count', 'Total background read count',
               'Cpf1 Indel Frequency', 'Cpf1 Indel read count', 
               'Cpf1 Total read count', 'No background indel frequency']
train_cpf1_df.columns = new_columns
test_cpf1_df.columns = new_columns
 # Alternatively, https://stackoverflow.com/questions/34263772/how-to-generate-one-hot-encoding-for-dna-sequences
  #list(string) breaks the string into a list. 
cols = [str(i) for i in range(20)]
train_single_nucleotides = pd.DataFrame()
test_single_nucleotides = pd.DataFrame()
for i in range(20):
  train_single_nucleotides[cols[i]] = [train_cpf1_df['20bp'][j][i] for j in range(15000)]
  test_single_nucleotides[cols[i]] = [test_cpf1_df['20bp'][j][i] for j in range(1292)]

train_labels = pd.get_dummies(train_single_nucleotides)
test_labels = pd.get_dummies(test_single_nucleotides)
train_targets = train_cpf1_df['No background indel frequency']
test_targets = test_cpf1_df['No background indel frequency']

#cv = KFold(n_splits=10)
reg = LinearRegression().fit(train_labels,train_targets)
pred = reg.predict(test_labels)
mse = metrics.mean_squared_error(test_targets,pred)
print('Mean Square Error: {:.3f}'.format(mse))



 

 
