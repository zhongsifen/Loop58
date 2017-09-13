#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# XOR
x_data = np.array([[[0], [0]], [[1], [0]], [[0], [1]], [[1], [1]]]).astype(np.float32)
y_data = np.array([[0], [1], [1], [0]]).astype(np.float32)

X = tf.placeholder(np.float32, shape=(2, 1))
Y = tf.placeholder(np.float32)

def model(X, r, c, w, b):
    h = tf.nn.relu(tf.matmul(r, X) + c)
    return tf.matmul(w, h) + b

_R = tf.constant([[1, 1], [1, 1]], dtype=tf.float32)
_C = tf.constant([[0], [-1]], dtype=tf.float32)
_W = tf.constant([[1, -2]], dtype=tf.float32)
_B = tf.constant([[0]], dtype=tf.float32)

r = tf.Variable(tf.random_normal(shape=(2, 2)))
c = tf.Variable(tf.random_normal(shape=(2, 1)))
w = tf.Variable(tf.random_normal(shape=(1, 2)))
b = tf.Variable(tf.zeros([1]))
#r = tf.Variable(_R)
#c = tf.Variable(_C)
#w = tf.Variable(_W)
#b = tf.Variable(_B)
y = model(X, r, c, w, b)

cost = tf.square(Y - y) # use square error for cost function
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    #_H = sess.run(model(x_data[3, :], _R, _C, _W, _B))
    for step in range(10000):
        for i in range(4):
            sess.run(train, feed_dict={X: x_data[i,:], Y: y_data[i,:]})
    R_ = sess.run(r)
    C_ = sess.run(c)
    W_ = sess.run(w)
    B_ = sess.run(b)
    H_ = [
            sess.run(model(x_data[0, :], R_, C_, W_, B_)),
            sess.run(model(x_data[1, :], R_, C_, W_, B_)),
            sess.run(model(x_data[2, :], R_, C_, W_, B_)),
            sess.run(model(x_data[3, :], R_, C_, W_, B_))
        ]       
#            print(step, sess.run(W), sess.run(b))
#            print(step, sess.run(R), sess.run(c), sess.run(W), sess.run(b))
#            for i in range(4):
#                print(sess.run(model(x_data[i,:], sess.run(R), sess.run(c), sess.run(W), sess.run(b))))
