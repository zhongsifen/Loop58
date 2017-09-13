#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:58:27 2017

@author: zhongsifen
"""

import tensorflow as tf
import numpy as np

#def model(X, w, b):
#    return tf.matmul(w, X) + b

def model(X, r, c, w, b):
    h = tf.nn.relu(tf.matmul(r, X) + c)
    return tf.matmul(w, h) + b

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
#A = np.array([211.5, 52.3])
#B = 10.4
#x_data = np.random.rand(2, 8).astype(np.float32)
#y_data = np.dot(A, x_data) + B

# XOR
x_data = np.array([[[0], [0]], [[1], [0]], [[0], [1]], [[1], [1]]]).astype(np.float32)
y_data = np.array([[0], [1], [1], [0]]).astype(np.float32)

#X = tf.placeholder("float")
#Y = tf.placeholder("float")
X = tf.placeholder(np.float32, shape=(2, 1))
Y = tf.placeholder(np.float32)

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
R = tf.Variable(tf.ones([2, 2]))
#c = tf.Variable(tf.constant([[0.0, -1.0], [0.0, -1.0], [0.0, -1.0], [0.0, -1.0]]))
c = tf.Variable(tf.constant([0.0, -1.0]))
W = tf.Variable(tf.constant([[1.0, -2.0]]))
b = tf.Variable(tf.zeros([1]))
y = model(X, R, c, W, b)

# Minimize the mean squared errors.
#loss = tf.reduce_mean(tf.square(y - Y))
#optimizer = tf.train.GradientDescentOptimizer(0.5)
#train = optimizer.minimize(loss)

cost = tf.square(Y - y) # use square error for cost function
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
#for step in range(2001):
#    sess.run(train)
#    if step % 20 == 0:
#        print(step, sess.run(R), sess.run(c), sess.run(W), sess.run(b))
#        print(sess.run(model(x_data, sess.run(R), sess.run(c), sess.run(W), sess.run(b))) - y_data)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.global_variables_initializer().run()

    for step in range(10000):
        for (x, y) in zip(x_data, y_data):
            sess.run(train, feed_dict={X: x, Y: y})
        if step % 1000 == 999:
#            print(step, sess.run(W), sess.run(b))
#            print(step, sess.run(R), sess.run(c), sess.run(W), sess.run(b))
            for i in range(4):
                print(sess.run(model(x_data[i,:], sess.run(R), sess.run(c), sess.run(W), sess.run(b))))
       
