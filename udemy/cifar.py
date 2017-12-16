import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

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
    batch_y = y_data[current_batch_index:current_batch_index + batch_size];
    for index, i in enumerate(batch_y):
        batch_y[index] = [0] * labels_count
        batch_y[index][i] = 1

    current_batch_index += batch_size
    return batch_X, batch_y;

def create_model():

    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.float32, [None, labels_count])

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

dict = unpickle('data_batch_1')
X_data = np.array(dict[b'data']).reshape(10000, 3, 32, 32).transpose([0, 2, 3, 1]).astype('uint8') / 255
y_data = dict[b'labels']

accuracies = []

with tf.Session() as sess:

    X, y, y_pred, model = create_model()

    sess.run(tf.global_variables_initializer())

    for epoch in range(10):

        current_batch_index = 0

        for step in range(10000//batch_size):
            batch_X, batch_y = next_batch()
            sess.run(model, feed_dict = { X: batch_X, y: batch_y })

            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(batch_y, 1), tf.argmax(y_pred, 1)), tf.float32))
            a = sess.run(acc, feed_dict = { X: batch_X, y: batch_y })
            accuracies.append(a)
            print(a)


    plt.plot(accuracies);
    plt.show()
