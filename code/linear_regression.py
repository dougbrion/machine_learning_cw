import tensorflow as tf
import helpers as hp
import numpy as np

def calc_error(_X, _y, _W, _b):
    pred = tf.add(tf.matmul(_X, _W), _b)
    cost = tf.reduce_mean(tf.square(_y - pred))
    return pred, cost

def linear_regression(_train_X, _train_y, _epochs, _rate):
    X = tf.placeholder(tf.float32, [None, hp.num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")

    W = tf.Variable(tf.random_normal([hp.num_features(_train_X), 1], dtype=tf.float32), name="weight")
    b = tf.Variable(tf.random_normal([1], dtype=tf.float32), name="bias")

    pred, cost = calc_error(X, y, W, b)
    
    optimizer = tf.train.GradientDescentOptimizer(_rate).minimize(cost)
    XyWb = [X, y, W, b]
    with tf.Session() as sess:
        # return hp.run(sess, XyWb, _train_X, _train_y, optimizer, cost, _epochs, _rate, "lin")
        return hp.cross_validation(sess, XyWb, _train_X, _train_y, optimizer, cost, _epochs, _rate, 10, "lin")

def run_linear_regression(epochs, rate):
    ds = hp.load_ds(hp.PATH, hp.FIXED)
    X, y = hp.split(ds)
    # return linear_regression(X, y, epochs, rate)
    linear_regression(X, y, epochs, rate)