import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
import sys
import os

os.system('clear')

batch_size = 50
batch_data = None 
labels_count = 0
current_batch_index = 0;

X = None;
y = None;

def unpickle(file):
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


def next_batch():
    global current_batch_index

    batch_X = X_data[current_batch_index:current_batch_index + batch_size];
    lab = y_data[current_batch_index:current_batch_index + batch_size];
    batch_y = np.zeros((batch_size, labels_count))
    for i, index in enumerate(lab):
        batch_y[i][index] = 1

    current_batch_index += batch_size
    return batch_X, batch_y;

def create_model():

    X = tf.Variable(tf.zeros([batch_size, 32, 32, 3]))
    y = tf.Variable(tf.zeros([batch_size, labels_count]))

    conv1 = tf.layers.conv2d(
            inputs = X,
            filters = 32,
            kernel_size = [5, 5],
            padding = 'same',
            activation = tf.nn.relu)

    flatten = tf.layers.flatten(inputs = conv1)

    dense = tf.layers.dense(inputs = flatten, units = 1024, activation = tf.nn.relu);

    y_pred = tf.layers.dense(inputs = dense, units = labels_count)

    # loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_pred))

    # optimizer
    optimizer = tf.train.AdamOptimizer()

    model = optimizer.minimize(cross_entropy);

    return X, y, y_pred, model

# -----------------------------------------[ init de boel ]------------------------------

meta = unpickle('batches.meta')
labels_count = len(meta[b'label_names'])

X_data = []
y_data = []
label_data = []


for i in range(5):
    dict = unpickle('data_batch_' + str(i + 1))
    a = np.array(dict[b'data']).reshape(10000, 3, 32, 32).transpose([0, 2, 3, 1]).astype('uint8') / 255
    b = dict[b'labels']

    X_data.append(a)
    label_data.append(b)

X_data = np.vstack(X_data)
label_data = np.ndarray.flatten(np.array(label_data))
y_data = np.zeros((len(label_data), labels_count));

for index, value in enumerate(label_data):
    y_data[index, value] = 1;

accuracies = []

total_batches = X_data.shape[0]//batch_size

with tf.Session() as sess:

    X, y, y_pred, model = create_model()

    sess.run(tf.global_variables_initializer())

    with tf.device('/gpu:0'):
        dataset_X = tf.constant(X_data)
        dataset_y = tf.constant(y_data)

        for epoch in range(10):

            index = 0

            current_batch_index = 0
            print("*" * 50)

            for step in range(total_batches):
                sys.stdout.write("Epoch %s => Batch: %d van %d   \r" % (epoch, step, total_batches))
                sys.stdout.flush()

                X = tf.slice(dataset_X, [index, 0, 0, 0], [batch_size, 32, 32, 3])
                y = batch_y = tf.slice(dataset_y, [index, 0], [batch_size, labels_count])
                sess.run(model)

                #acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(batch_y, 1), tf.argmax(y_pred, 1)), tf.float32))
                #a = sess.run(acc, feed_dict = { X: batch_X, y: batch_y })
                #accuracies.append(a)

                tf.get_variable_scope().reuse_variables() 
                kernel = tf.get_variable('conv2d/kernel');
                #w_min = tf.reduce_min(kernel);
                #w_max = tf.reduce_max(kernel);
                #kernel = (kernel - w_min) / (w_max - w_min);
                weights = sess.run(kernel)
                weights = weights.transpose([3, 0, 1, 2])
                f = open('weights.json', 'w')
                f.write(json.dumps({
                    'weights': weights.tolist()
                }))
                f.close()
