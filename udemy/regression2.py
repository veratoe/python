import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

import os

from sklearn.model_selection import train_test_split

pd.set_option('display.width', 1000)
os.system('clear')

diabetes = pd.read_csv('pima-indians-diabetes.csv')

cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps', 'Insulin', 'BMI', 'Pedigree']

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

#assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', diabetes['Group'].unique())

age_bucket = tf.feature_column.bucketized_column(age, boundaries = [ 20, 30, 40, 50, 60, 70, 80])

#feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, assigned_group, age_bucket]

#train test split
x_data = diabetes.drop('Class', axis = 1)
labels = diabetes['Class']

X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size = 0.3, random_state = 101)

#input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size = 10, num_epochs = 1000, shuffle = True)
#model = tf.estimator.LinearClassifier(feature_columns = feat_cols, n_classes = 2)
#model.train(input_fn = input_func, steps = 1000)
#print('klaar met trainen');
#
#eval_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, y = y_test, batch_size = 10, num_epochs = 1, shuffle = False)
#
#results = model.evaluate(eval_input_func);
#print()
#print(results)
#
#pred_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size = 10, num_epochs = 1, shuffle = False)
#my_predictions = list(model.predict(pred_input_func))

# DNN
assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size = 10)
embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension = 4)
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, embedded_group_col, diabetes_pedigree, age_bucket]
input_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size = 10, num_epochs = 1000, shuffle = True)
dnn_model = tf.estimator.DNNClassifier(hidden_units = [10, 20, 20, 20, 20], feature_columns = feat_cols, n_classes = 2)

dnn_model.train(input_fn = input_func, steps = 10000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, y = y_test, batch_size = 10, num_epochs = 1, shuffle = False)

results = dnn_model.evaluate(eval_input_func);
print("DE RESULTATEN:")
print("-" * 50)
print(results)
