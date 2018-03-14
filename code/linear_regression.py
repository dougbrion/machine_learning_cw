import tensorflow as tf
import helpers as hp
import numpy as np

def calc_error(_X, _y, _W, _b):
    pred = tf.add(tf.matmul(_X, _W), _b)
    cost = tf.reduce_mean(tf.square(_y - pred))
    return pred, cost

def calc_error_L1(_X, _y, _W, _b, _scale=0.005):
    L1 = tf.contrib.layers.l1_regularizer(scale = _scale)
    reg_cost = tf.contrib.layers.apply_regularization(L1, [_W])
    pred, cost = calc_error(_X, _y, _W, _b)
    cost += reg_cost
    return pred, cost

def calc_error_L2(_X, _y, _W, _b, _scale=0.005):
    L2 = tf.contrib.layers.l2_regularizer(scale = _scale)
    reg_cost = tf.contrib.layers.apply_regularization(L2, [_W])
    pred, cost = calc_error(_X, _y, _W, _b)
    cost += reg_cost
    return pred, cost

def linear_regression(_train_X, _train_y, _test_X, _test_y, _epochs=1000, _rate=0.001, _regularisation=0, _cross_val=False):
    X = tf.placeholder(tf.float32, [None, hp.num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")

    W = tf.Variable(tf.random_normal([hp.num_features(_train_X), 1], dtype=tf.float32), name="weight")
    b = tf.Variable(tf.random_normal([1], dtype=tf.float32), name="bias")

    print("Regularisation: ", _regularisation)
    if _regularisation == 1:
        pred, cost = calc_error_L1(X, y, W, b)
    elif _regularisation == 2:
        pred, cost = calc_error_L2(X, y, W, b)
    else:
        pred, cost = calc_error(X, y, W, b)

    optimizer = tf.train.GradientDescentOptimizer(_rate).minimize(cost)
    XyWb = [X, y, W, b]
    with tf.Session() as sess:
        if _cross_val == True:
            return hp.cross_validation(sess, XyWb, _train_X, _train_y, _test_X, _test_y, optimizer, cost, _epochs, _rate, "lin")
        else:
            return hp.run(sess, XyWb, _train_X, _train_y, _test_X, _test_y, optimizer, cost, _epochs, _rate, "lin")
