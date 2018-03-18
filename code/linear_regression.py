import tensorflow as tf
import helpers as hp
import numpy as np

def elastic_error(_X, _y, _W, _b):
    print("Elastic Net Loss Function")
    pred = tf.add(tf.matmul(_X, _W), _b)
    elastic_param1 = tf.constant(1.)
    elastic_param2 = tf.constant(1.)
    l1_a_loss = tf.reduce_mean(tf.abs(_W))
    l2_a_loss = tf.reduce_mean(tf.square(_W))
    e1_term = tf.multiply(elastic_param1, l1_a_loss)
    e2_term = tf.multiply(elastic_param2, l2_a_loss)
    cost = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(_y - pred)), e1_term), e2_term), 0)    
    return pred, cost

def svm_error(_X, _y, _W, _b):
    print("SVM Loss Function")
    pred = tf.add(tf.matmul(_X, _W), _b)
    epsilon = tf.constant([0.5])
    cost = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(pred, _y)), epsilon)))
    return pred, cost

# L1 cost function 
def calc_error_L1(_X, _y, _W, _b):
    print("Loss Function L1")
    pred = tf.add(tf.matmul(_X, _W), _b)
    cost = tf.reduce_mean(tf.abs(_y - pred))
    return pred, cost

# L2 cost function
def calc_error_L2(_X, _y, _W, _b):
    print("Loss Function L2")
    pred = tf.add(tf.matmul(_X, _W), _b)
    cost = tf.reduce_mean(tf.square(_y - pred))
    return pred, cost

def calc_error(_X, _y, _W, _b, _cost_fn):
    if _cost_fn == 1:
        pred, cost = calc_error_L1(_X, _y, _W, _b) 
    elif _cost_fn == 2:
        pred, cost = calc_error_L2(_X, _y, _W, _b)
    elif _cost_fn == 3:
        pred, cost = elastic_error(_X, _y, _W, _b)
    elif _cost_fn == 4:
        pred, cost = svm_error(_X, _y, _W, _b)
    return pred, cost

def calc_error_reg_L1(_X, _y, _W, _b, cost_fn, _scale):
    print("Regularisation L1")
    L1 = tf.contrib.layers.l1_regularizer(scale = _scale)
    reg_cost = tf.contrib.layers.apply_regularization(L1, [_W])
    pred, cost = calc_error(_X, _y, _W, _b, cost_fn)
    cost += reg_cost
    return pred, cost

def calc_error_reg_L2(_X, _y, _W, _b, cost_fn, _scale):
    print("Regularisation L2")
    L2 = tf.contrib.layers.l2_regularizer(scale = _scale)
    reg_cost = tf.contrib.layers.apply_regularization(L2, [_W])
    pred, cost = calc_error(_X, _y, _W, _b, cost_fn)
    cost += reg_cost
    return pred, cost

def linear_regression(_train_X, _train_y, _test_X, _test_y, _epochs, _rate, _cost_fn, _regularisation, _cross_val):
    reg_type, reg_scale = _regularisation
    X = tf.placeholder(tf.float32, [None, hp.num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")

    W = tf.Variable(tf.random_normal([hp.num_features(_train_X), 1], dtype=tf.float32), name="weight")
    b = tf.Variable(tf.random_normal([1], dtype=tf.float32), name="bias")

    if reg_type == 1:
        pred, cost = calc_error_reg_L1(X, y, W, b, _cost_fn, reg_scale)
    elif reg_type == 2:
        pred, cost = calc_error_reg_L2(X, y, W, b, _cost_fn, reg_scale)
    else:
        print("No Regularisation")
        pred, cost = calc_error(X, y, W, b, _cost_fn)

    optimizer = tf.train.GradientDescentOptimizer(_rate).minimize(cost)
    XyWb = [X, y, W, b]
    with tf.Session() as sess:
        if _cross_val == True:
            return hp.cross_validation(sess, XyWb, _train_X, _train_y, _test_X, _test_y, optimizer, cost, _epochs, _rate, "lin")
        else:
            return hp.run(sess, XyWb, _train_X, _train_y, _test_X, _test_y, optimizer, cost, _epochs, _rate, "lin")
