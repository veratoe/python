import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

import os

from sklearn.model_selection import train_test_split

pd.set_option('display.width', 1000)
os.system('clear')

housing = pd.read_csv('cal_housing_clean.csv')

X = housing.drop('medianHouseValue', axis = 1)
y = housing['medianHouseValue']

# normalize columns
cols = ['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'households', 'medianIncome']
X[cols] = X[cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

feature_cols = []

for col in cols:
    feature_cols.append(tf.feature_column.numeric_column(col))

input_fn = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size = 10, shuffle = True, num_epochs = 1000)
model = tf.estimator.DNNRegressor(hidden_units = [10, 20, 20, 20, 10], feature_columns = feature_cols)
tf.logging.set_verbosity(tf.logging.INFO)
print('training ...')
model.train(input_fn = input_fn, steps = 100000)
print('done! ...')

prediction_fn = tf.estimator.inputs.pandas_input_fn(X_test, y_test, batch_size = 10, shuffle = False, num_epochs = 1 )
print('predicting..')
predictions = model.predict(prediction_fn)
print('done!')

p = []
error = 0
for row, prediction in zip(list(y_test), predictions):
    error += (row - prediction['predictions'][0]) ** 2
    p.append(prediction['predictions'][0])

print('RMSE: %s' % (error / len(y_test)) ** 0.5)
input()

#evaluation = model.evaluate(prediction_fn)
#print('done!');
#print('=' * 50)
#print(evaluation)

