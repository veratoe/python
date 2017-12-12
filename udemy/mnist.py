import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
import numpy as np

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape = shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    # x = input tensor [batch, H, W, Channels]
    # W = [filter H, filter W, Channels In, Channels, Out ]
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2by2(x):
    # x = input tensor [batch, H, W, Channels]
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normal_layer(input_layer, size):
     input_size = int(input_layer.get_shape()[1])
     W = init_weights([input_size, size])
     b = init_bias([size])
     return tf.matmul(input_layer, W) + b


def create_model():
    # placeholders
    x = tf.placeholder(tf.float32, shape = [None, 784])
    y_true = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    convo_1 = convolutional_layer(x_image, shape = [5, 5, 1, 32])
    pool_1 = max_pool_2by2(convo_1)

    convo_2 = convolutional_layer(pool_1, shape = [5, 5, 32, 64])
    pool_2 = max_pool_2by2(convo_2)

    flat = tf.reshape(pool_2, [-1, 7*7*64])
    full_layer = tf.nn.relu(normal_layer(flat, 1024)) 

    hold_prob = tf.placeholder(tf.float32)
    dropout = tf.nn.dropout(full_layer, keep_prob = hold_prob)

    y_pred = normal_layer(dropout, 10)

    # loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred))

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
    train = optimizer.minimize(cross_entropy);

    return x, y_pred, y_true, train, hold_prob

def train():

    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    x, y_pred, y_true, train, hold_prob = create_model()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        accuracies = []
        init = tf.global_variables_initializer()
        sess.run(init)

        os.system('clear')
        print("Trainen...")

        for step in range(1000):
            batch_x, batch_y = mnist.train.next_batch(50)
            sess.run(train, feed_dict={ x: batch_x, y_true: batch_y, hold_prob: 0.5 })

            if step % 10 == 0:
                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
                acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                acc_score = sess.run(acc, feed_dict = { x: mnist.test.images, y_true: mnist.test.labels, hold_prob: 1.0 })
                accuracies.append(acc_score)
                print("De accuracy na stap " + str(step) + " is: " + str(acc_score))

        saver.save(sess, "./model.ckpt")

        plt.plot(accuracies)
        plt.show()


def predict():

    from PIL import Image

    tf.reset_default_graph()

    x, y_pred, y_true, train, hold_prob = create_model()
    saver = tf.train.Saver()

    data = np.array(Image.open("sample.png"))
    data = np.sum(data, 2) / 255
    data = np.reshape(data, (1, 784))

    with tf.Session() as sess:

        saver.restore(sess, "./model.ckpt")

        predict = sess.run(y_pred, feed_dict={ x: data, hold_prob: 1 })
        print(predict)

    return predict
