import tensorflow as tf
import pandas as pd
import helpers as hp
import numpy as np

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
    # X = _ds.drop(['quality'], axis=1).values
    # y = _ds['quality'].values
    # return X, y

def num_features(_ds):
    return _ds.shape[1]

def calc_error(_X, _y, _W, _b):
    pred = tf.add(tf.matmul(_X, _W), _b)
    cost = tf.reduce_mean(tf.square(_y - pred))
    return pred, cost

def linear_regression(_train_X, _train_y):
    X = tf.placeholder(tf.float32, [None, num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")

    W = tf.Variable(tf.random_normal([num_features(_train_X), 1], dtype=tf.float32), name="weight")
    b = tf.Variable(tf.random_normal([1], dtype=tf.float32), name="bias")

    pred, cost = calc_error(X, y, W, b)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    merged_summaries = tf.summary.merge_all()
    with tf.Session() as sess:

        log_directory = 'tmp/logs'
        summary_writer = tf.summary.FileWriter(log_directory, sess.graph)
        sess.run(init)

        cost_sum = 0
        for epoch in range(epochs):
            # for (xi, yi) in zip(_train_X, _train_y):
            _, c = sess.run([optimizer, cost], feed_dict={X: _train_X, y: _train_y})

            cost_sum += c

            if (epoch + 1) % step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))
            

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: _train_X, y: _train_y})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

ds = load_ds(hp.PATH, hp.FIXED)
X, y = split(ds)
linear_regression(X, y)