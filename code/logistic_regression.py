import tensorflow as tf
import numpy as np
import helpers as hp

def calc_error(_X, _y, _W, _b):
    pred = tf.add(tf.matmul(_X, _W), _b)
    # cost = tf.reduce_mean(tf.reduce_sum(tf.matmul(tf.transpose(_y), tf.log(pred)), reduction_indices=1))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=_y))
    return pred, cost

def logistic_regression(_train_X, _train_y, _epochs, _rate):
    X = tf.placeholder(tf.float32, [None, hp.num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")

    W = tf.Variable(tf.zeros([hp.num_features(_train_X), 1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    # W = tf.Variable(tf.random_normal([hp.num_features(_train_X), 1], dtype=tf.float32), name="weight")
    # b = tf.Variable(tf.random_normal([1], dtype=tf.float32), name="bias")

    _, cost = calc_error(X, y, W, b)

    optimizer = tf.train.GradientDescentOptimizer(_rate).minimize(cost)
    XyWb = [X, y, W, b]


    with tf.Session() as sess:
        return hp.run(sess, XyWb, _train_X, _train_y, optimizer, cost, _epochs, _rate, "log")

def run_logistic_regression(epochs, rate):
    ds = hp.load_ds(hp.PATH, hp.FIXED)
    X, y = hp.split(ds)
    return logistic_regression(X, y, epochs, rate)