import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# placeholders
x = tf.placeholder(tf.float32, shape = [None, 784])

# variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# create graph operations
y = tf.matmul(x, W) + b

# loss function
y_true = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)
train = optimizer.minimize(cross_entropy);

init = tf.global_variables_initializer()


os.system('clear')

accuracies = []

with tf.Session() as sess:

    sess.run(init)

    print("Trainen...")

    for step in range(50):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={ x: batch_x, y_true: batch_y })

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc_score = sess.run(acc, feed_dict = { x: mnist.test.images, y_true: mnist.test.labels })
        accuracies.append(acc_score)
        print("De accuracy na stap " + str(step) + " is: " + str(acc_score))

    # evaluate
    plt.plot(accuracies, 'c') 
    plt.show()

    for i in random.sample(range(1, mnist.train.images.shape[0]), 20):
        prediction = sess.run(y, feed_dict = { x: [mnist.train.images[i]] })
        tops, indices = sess.run(tf.nn.top_k(prediction, k = 2))
        plt.imshow(mnist.train.images[i].reshape(28, 28))
        plt.title('TF denkt dat dit een ' + str(indices[0, 0]) + ' is met s = ' + str(tops[0, 0]) + ' of ' + str(indices[0, 1]) + ' met s = ' + str(tops[0, 1]))
        plt.show()

    input()


