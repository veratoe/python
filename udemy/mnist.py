import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
import numpy as np

def create_model():
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

    return x, y, y_true, train

def train():

    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    x, y, y_true, train = create_model()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        accuracies = []
        init = tf.global_variables_initializer()
        sess.run(init)

        os.system('clear')
        print("Trainen...")

        for step in range(50):
            batch_x, batch_y = mnist.train.next_batch(100)
            sess.run(train, feed_dict={ x: batch_x, y_true: batch_y })

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            acc_score = sess.run(acc, feed_dict = { x: mnist.test.images, y_true: mnist.test.labels })
            accuracies.append(acc_score)
            print("De accuracy na stap " + str(step) + " is: " + str(acc_score))

        saver.save(sess, "./model.ckpt")

def predict():

    from PIL import Image

    tf.reset_default_graph()

    x, y, y_true, train = create_model()
    saver = tf.train.Saver()

    data = np.array(Image.open("sample.png"))
    data = np.sum(data, 2) / 255
    data = np.reshape(data, (1, 784))

    with tf.Session() as sess:

        saver.restore(sess, "./model.ckpt")

        predict = sess.run(y, feed_dict={ x: data })
        print(predict)

    return predict
