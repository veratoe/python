import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
import sys
import os
import random
from sklearn.model_selection import train_test_split

os.system('clear')

batch_size = 64
batch_data = None 
labels_count = 0
current_batch_index = 0;

is_training = True

X = None;
y = None;

def unpickle(file):
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


def next_batch(X, y):
    global current_batch_index

    batch_X = X[current_batch_index:current_batch_index + batch_size];
    batch_y = y[current_batch_index:current_batch_index + batch_size];
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

    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2)

    conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 64,
            kernel_size = [5, 5],
            padding = 'same',
            activation = tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)

    flattened_pool2 = tf.reshape(pool2, [-1, 8 * 8 * 64])


    dense = tf.layers.dense(inputs = flattened_pool2, units = 1024, activation = tf.nn.relu);

    dropout = tf.layers.dropout(inputs = dense, rate = 0.4, training = is_training)

    y_pred = tf.layers.dense(inputs = dropout, units = 10)

    # loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_pred))

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)

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

#for b in range(20):
#    i = random.randrange(50000)
#    plt.imshow(X_data[i])
#    plt.title(meta[b'label_names'][np.argmax(y_data[i])]);
#    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.33, random_state = 102)
accuracies = []

total_batches = X_train.shape[0]//batch_size

with tf.Session() as sess:

    X, y, y_pred, model = create_model()

    sess.run(tf.global_variables_initializer())

    for epoch in range(100):

        current_batch_index = 0
        print("*" * 50)

        for step in range(total_batches):
            sys.stdout.write("Epoch %s => Batch: %d van %d   \r" % (epoch, step, total_batches))
            sys.stdout.flush()

            batch_X, batch_y = next_batch(X_train, y_train)
            is_training = True;
            sess.run(model, feed_dict = { X: batch_X, y: batch_y })


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

        is_training = False
        acc = tf.equal(tf.argmax(y_test, 1), tf.argmax(y_pred, 1));
        results = sess.run(acc, feed_dict = { X: X_test, y: y_test })
        print("\n")
        print(results)
        print(np.sum(results) / len(results))
        #accuracies.append(a)
        #print("Acc is: %s" % a);
        #input()
