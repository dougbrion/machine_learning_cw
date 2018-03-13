import tensorflow as tf
import numpy as np
import helpers as hp
import pandas as pd
import matplotlib.pyplot as plt

learning_rate = 0.00001
epochs = 1000
step = 50

def load_ds(_path, _infile):
    ds = pd.read_csv(_path + _infile, sep = ',')
    return ds

def split(_ds):
    data = _ds.values
    split = np.split(data, [11], axis=1)
    return split[0], split[1]

def calc_error(_X, _y, _W, _b):
    pred = tf.add(tf.matmul(_X, _W), _b)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=_y))
    return pred, cost

def logistic_regression(_train_X, _train_y):
    X = tf.placeholder(tf.float32, [None, hp.num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")

    W = tf.Variable(tf.zeros([hp.num_features(_train_X), 1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    # W = tf.Variable(tf.random_normal([hp.num_features(_train_X), 1], dtype=tf.float32), name="weight")
    # b = tf.Variable(tf.random_normal([1], dtype=tf.float32), name="bias")

    _, cost = calc_error(X, y, W, b)

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
                print("Epoch:", '%04d' % (epoch + 1), "cost=", c)
            
        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: _train_X, y: _train_y})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
        # plt.plot(_train_X, _train_y, 'ro', label='Original Data', marker='.')
        # plt.legend()
        # plt.show()
        plt.plot(x_axis, y_axis)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Training Error')
        plt.grid(linestyle='-')
        plt.show()


ds = load_ds(hp.PATH, hp.FIXED)
X, y = split(ds)
logistic_regression(X, y)