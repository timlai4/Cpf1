# Copyright 2017 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 23:28:39 2019

@author: Tim
"""




from __future__ import print_function

import math

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
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

train_single_counts = pd.DataFrame()
test_single_counts = pd.DataFrame()
A_cols = []
C_cols = []
G_cols = []
T_cols = []
for i in range(34):
  A_cols.append(str(i) + "_A")
  C_cols.append(str(i) + "_C")  
  G_cols.append(str(i) + "_G")
  T_cols.append(str(i) + "_T")
  
del A_cols[4:7] # Training set does not have A's in these positions.
del C_cols[4:7]
del G_cols[4:7]

train_single_counts['A'] = (train_single_encode[A_cols].sum(axis=1))/34
train_single_counts['C'] = (train_single_encode[C_cols].sum(axis=1))/34
train_single_counts['G'] = (train_single_encode[G_cols].sum(axis=1))/34
train_single_counts['T'] = (train_single_encode[T_cols].sum(axis=1))/34

test_single_counts['A'] = (test_single_encode[A_cols].sum(axis=1))/34
test_single_counts['C'] = (test_single_encode[C_cols].sum(axis=1))/34
test_single_counts['G'] = (test_single_encode[G_cols].sum(axis=1))/34
test_single_counts['T'] = (test_single_encode[T_cols].sum(axis=1))/34

train_labels= pd.concat([train_single_encode, train_di_encode,train_single_counts], axis = 1)
test_labels = pd.concat([test_single_encode, test_di_encode,test_single_counts], axis = 1)

train_targets = pd.DataFrame({'No background indel frequency': train_cpf1_df['No background indel frequency']})
test_targets = pd.DataFrame({'No background indel frequency': test_cpf1_df['No background indel frequency']})

# Free up some memory
del train_single_nucleotides
del test_single_nucleotides
del train_dinucleotides
del test_dinucleotides
del train_single_encode
del test_single_encode
del train_di_encode
del test_di_encode

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a neural network model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(15000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])
      
      
def train_nn_regression_model(
    my_optimizer,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a neural network regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    my_optimizer: An instance of `tf.train.Optimizer`, the optimizer to use.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      
  Returns:
    A tuple `(estimator, training_losses, validation_losses)`:
      estimator: the trained `DNNRegressor` object.
      training_losses: a `list` containing the training loss values taken during training.
      validation_losses: a `list` containing the validation loss values taken during training.
  """

  periods = 10
  steps_per_period = steps / periods
  
  # Create a DNNRegressor object.
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_regressor = tf.estimator.DNNRegressor(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer
  )
  
  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets['No background indel frequency'], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets['No background indel frequency'], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets['No background indel frequency'], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
  print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

  return dnn_regressor, training_rmse, validation_rmse

_ = train_nn_regression_model(
    my_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.007,l1_regularization_strength=0.005, l2_regularization_strength = 0.005),
    steps=7000,
    batch_size=250,
    hidden_units=[2000, 3000, 3000, 3000, 3000, 2000],
    training_examples=train_labels,
    training_targets=train_targets,
    validation_examples=test_labels,
    validation_targets=test_targets)
