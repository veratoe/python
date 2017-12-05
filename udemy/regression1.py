import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

import os

os.system('clear')

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))
_m = np.random.rand()
_b = np.random.rand() * 10
y_true = (_m * x_data) + _b + noise;


print('We beginnen met m = %s, b = %s' % (_m, _b))
print()

x_df = pd.DataFrame(data = x_data)
y_df = pd.DataFrame(data = y_true)
my_data = pd.concat([x_df, y_df], axis = 1)

batch_size = 128 

m = tf.Variable(np.random.randn());
b = tf.Variable(np.random.randn());

xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32,  [batch_size])

y_model = m * xph + b

error = tf.reduce_sum(tf.square(yph - y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0002)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

bs = []
ms = []
e = []

with tf.Session() as sess:

    sess.run(init)
    batches = 250

    for i in range(batches):
        # random index
        index = np.random.randint(len(x_data - batch_size - 1), size = batch_size)
        feed = { xph: x_data[index], yph: y_true[index] }
        sess.run(train, feed_dict = feed)
        e.append(sess.run(error, feed_dict = feed))
        ms.append(sess.run(m))
        bs.append(sess.run(b))

    result = sess.run([m, b])

    print()
    print('Model is getraind, uitkomsten: ');
    print(result)

    print('Accuracy: %0.2f, %0.2f' % (result[0]/ _m, result[1] / _b)) 

sample = my_data.sample(250).as_matrix()


plt.scatter(sample[:, 0], sample[:, 1])
x = np.arange(11)
y = x * result[0] + result[1] 
plt.plot(x, y, 'red')
plt.show()

fig, f1 = plt.subplots()
f2 = f1.twinx()

f1.plot(bs, 'blue')
f1.plot(ms, 'green')
f2.plot(e, 'brown')
plt.show()
