import tensorflow as tf
import numpy as np
import helpers as hp

def calc_error(_X, _y, _W, _b):
    pred = tf.add(tf.matmul(_X, _W), _b)
    # cost = tf.reduce_mean(tf.reduce_sum(tf.matmul(tf.transpose(_y), tf.log(pred)), reduction_indices=1))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=_y))
    return pred, cost

def calc_error_L1(_X, _y, _W, _b, _scale=0.1):
    L1 = tf.contrib.layers.l1_regularizer(scale = _scale)
    reg_cost = tf.contrib.layers.apply_regularization(L1, [_W])
    pred, cost = calc_error(_X, _y, _W, _b)
    cost += reg_cost
    return pred, cost

def calc_error_L2(_X, _y, _W, _b, _scale=0.1):
    L2 = tf.contrib.layers.l2_regularizer(scale = _scale)
    reg_cost = tf.contrib.layers.apply_regularization(L2, [_W])
    pred, cost = calc_error(_X, _y, _W, _b)
    cost += reg_cost
    return pred, cost

def logistic_regression(_train_X, _train_y, _test_X, _test_y, _epochs, _rate,  _regularisation, _cross_val):
    reg_type, reg_scale = _regularisation
    X = tf.placeholder(tf.float32, [None, hp.num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")

    W = tf.Variable(tf.zeros([hp.num_features(_train_X), 1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")

    print("Regularisation: ", reg_type)
    if reg_type == 1:
        pred, cost = calc_error_L1(X, y, W, b, reg_scale)
    elif reg_type == 2:
        pred, cost = calc_error_L2(X, y, W, b, reg_scale)
    else:
        pred, cost = calc_error(X, y, W, b)

    optimizer = tf.train.GradientDescentOptimizer(_rate).minimize(cost)
    XyWb = [X, y, W, b]

    with tf.Session() as sess:
        if _cross_val == True:
            return hp.cross_validation(sess, XyWb, _train_X, _train_y, _test_X, _test_y, optimizer, cost, _epochs, _rate, "log")
        else:
            return hp.run(sess, XyWb, _train_X, _train_y, _test_X, _test_y, optimizer, cost, _epochs, _rate, "log")
