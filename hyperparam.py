# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:30:50 2019

@author: Tim
"""


import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

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
cols = [str(i) for i in range(34)]
train_single_nucleotides = pd.DataFrame()
test_single_nucleotides = pd.DataFrame()
train_dinucleotides = pd.DataFrame()
test_dinucleotides = pd.DataFrame()
for i in range(34):
  train_single_nucleotides[cols[i]] = [train_cpf1_df['34bp'][j][i] for j in range(15000)]
  test_single_nucleotides[cols[i]] = [test_cpf1_df['34bp'][j][i] for j in range(1292)]

for i in range(32):
  train_dinucleotides[cols[i] + cols[i]] = [train_cpf1_df['34bp'][j][i:i+2] for j in range(15000)]
  test_dinucleotides[cols[i] + cols[i]] = [test_cpf1_df['34bp'][j][i:i+2] for j in range(1292)]

train_single_encode = pd.get_dummies(train_single_nucleotides)
test_single_encode = pd.get_dummies(test_single_nucleotides)

train_di_encode = pd.get_dummies(train_dinucleotides)
test_di_encode = pd.get_dummies(test_dinucleotides)

train_labels= pd.concat([train_single_encode, train_di_encode], axis = 1)
test_labels = pd.concat([test_single_encode, test_di_encode], axis = 1)
# Free up some memory
del train_single_nucleotides
del test_single_nucleotides
del train_dinucleotides
del test_dinucleotides
del train_single_encode
del test_single_encode
del train_di_encode
del test_di_encode

train_targets = train_cpf1_df['No background indel frequency']
test_targets = test_cpf1_df['No background indel frequency']

processed_features = pd.concat([train_labels, test_labels], ignore_index = True, sort = False)
processed_targets = pd.concat([train_targets, test_targets], ignore_index = True, sort = False)
reg = MLPRegressor(hidden_layer_sizes=(2000, 3000, 3000, 3000, 2000),max_iter = 2500, early_stopping = True)

parameters = { 
	'alpha': [0.1,0.01],
    'batch_size': [200,500,1000]
	}

CV_clf = GridSearchCV(estimator = reg, param_grid = parameters, scoring = 'mean_squared_error', cv = 3)
CV_clf.fit(processed_features, processed_targets)
print(CV_clf.best_params_)



