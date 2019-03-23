# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 23:54:17 2019

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

train_labels= pd.concat([train_single_encode, train_di_encode], axis = 1)
test_labels = pd.concat([test_single_encode, test_di_encode], axis = 1)

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
    """Trains a linear regression model of multiple features.
  
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
    
def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model of multiple features.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `suicide_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `suicide_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `suicide_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `suicide_dataframe` to use as target for validation.
      
  Returns:
    A `LinearRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods
  
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )
  
  # Create input functions.
  training_input_fn = lambda: my_input_fn(
      training_examples, 
      training_targets['No background indel frequency'], 
      batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(
      training_examples, 
      training_targets['No background indel frequency'], 
      num_epochs=1, 
      shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(
      validation_examples, validation_targets['No background indel frequency'], 
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
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    # Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
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

  return linear_regressor

linear_regressor = train_model(
    learning_rate=0.01,
    steps=2000,
    batch_size=50,
    training_examples=train_labels,
    training_targets=train_targets,
    validation_examples=test_labels,
    validation_targets=test_targets)