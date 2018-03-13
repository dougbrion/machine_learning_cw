import tensorflow as tf
import pandas as pd
import helpers as hp
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.001
epochs = 100
step = 50
BATCH_SIZE = 100

def calc_error(_X, _y, _W, _b):
    pred = tf.add(tf.matmul(_X, _W), _b)
    regularization_loss = 0.5 * tf.reduce_sum(tf.square(W)) 
    hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE,1]), 1 - y * pred));
    cost = regularization_loss + hinge_loss;
    return pred, cost

def support_vector_machine(_train_X, _train_y):
    X = tf.placeholder(tf.float32, [None, hp.num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")

    W = tf.Variable(tf.random_normal([hp.num_features(_train_X), 1], dtype=tf.float32), name="weight")
    b = tf.Variable(tf.random_normal([1], dtype=tf.float32), name="bias")

    pred, cost = calc_error(X, y, W, b)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    merged_summaries = tf.summary.merge_all()
    x_axis, y_axis = [], []
    with tf.Session() as sess:

        log_directory = 'tmp/logs'
        summary_writer = tf.summary.FileWriter(log_directory, sess.graph)
        sess.run(init)

        for epoch in range(epochs):
            _, c = sess.run([optimizer, cost], feed_dict={X: _train_X, y: _train_y})

            if (epoch % 10) == 0:
                x_axis.append(epoch+1)
                y_axis.append(c)

            if (epoch + 1) % step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))
            

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: _train_X, y: _train_y})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
        # plt.plot(_train_X, _train_y, 'ro', marker='.')
        # plt.title('XY plot of Original Data')
        # plt.xlabel('X')
        # plt.ylabel('y')
        # # plt.plot(_train_X, np.dot(_train_X, sess.run(W)) + sess.run(b), label='Fitted line')
        # plt.show()

        plt.plot(x_axis, y_axis)
        plt.title('Basic Linear Regression, Epochs=100, Learning Rate=0.001')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Training Error')
        plt.grid(linestyle='-')
        plt.show()

def run_support_vector_machine():
    ds = hp.load_ds(hp.PATH, hp.FIXED)
    X, y = hp.split(ds)
    support_vector_machine(X, y)