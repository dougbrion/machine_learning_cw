import tensorflow as tf
import helpers as hp
import numpy as np
import sys

def huber_error(_X, _y, _W, _b, _delta=1.0):
    pred = tf.add(tf.matmul(_X, _W), _b)
    residual = tf.abs(_y - pred)
    cond = tf.less(residual, _delta)
    small_res = 0.5 * tf.square(residual)
    large_res = _delta * residual - 0.5 * tf.square(_delta)
    cost = tf.reduce_mean(tf.where(cond, small_res, large_res))
    return pred, cost

def elastic_error(_X, _y, _W, _b, _param1=1., _param2=1.):
    print("Elastic Net Loss Function")
    pred = tf.add(tf.matmul(_X, _W), _b)
    param1 = tf.constant(_param1)
    param2 = tf.constant(_param2)
    print("Elastic Net Parameters are p1 ", _param1, _param2)
    l1loss = tf.reduce_mean(tf.abs(_W))
    l2loss = tf.reduce_mean(tf.square(_W))
    elastic1 = tf.multiply(param1, l1loss) 
    elastic2 = tf.multiply(param2, l2loss)
    cost = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(_y - pred)), elastic1), elastic2), 0)    
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

def calc_error(_X, _y, _W, _b, _cost_fn, _el_params=[1.,1.]):
    if _cost_fn == 1:
        pred, cost = calc_error_L1(_X, _y, _W, _b) 
    elif _cost_fn == 2:
        pred, cost = calc_error_L2(_X, _y, _W, _b)
    elif _cost_fn == 3:
        p1, p2 = _el_params
        pred, cost = elastic_error(_X, _y, _W, _b, p1, p2)
    elif _cost_fn == 4:
        pred, cost = svm_error(_X, _y, _W, _b)
    elif _cost_fn == 5:
        pred, cost = huber_error(_X, _y, _W, _b)
    else:
        pred, cost = calc_error_L1(_X, _y, _W, _b) 
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

def calc_error_reg_elastic(_X, _y, _W, _b, cost_fn, _el_params=[1.,1.]):
    print("Regularisation Elastic Net")
    lamb, alpha = _el_params
    l1 = tf.multiply(alpha, tf.norm(_W))
    l2 = tf.multiply((1.0 - alpha), tf.square(tf.norm(_W)))
    reg_cost = lamb * (l1 + l2)
    pred, cost = calc_error(_X, _y, _W, _b, cost_fn)
    cost += reg_cost
    return pred, cost

def linear_regression(_train_X, _train_y, _test_X, _test_y, _epochs, _rate, _cost_fn, _regularisation, _cross_val, _el_params=[1.,1.]):
    reg_type, reg_scale = _regularisation
    X = tf.placeholder(tf.float32, [None, hp.num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")

    W = tf.Variable(tf.random_normal([hp.num_features(_train_X), 1], dtype=tf.float32), name="weight")
    b = tf.Variable(tf.random_normal([1], dtype=tf.float32), name="bias")

    _, huber_cost = huber_error(X, y, W, b)
    _, lad = calc_error_L1(X, y, W, b)

    if reg_type == 1:
        pred, cost = calc_error_reg_L1(X, y, W, b, _cost_fn, reg_scale)
    elif reg_type == 2:
        pred, cost = calc_error_reg_L2(X, y, W, b, _cost_fn, reg_scale)
    elif reg_type == 3:
        pred, cost = calc_error_reg_elastic(X, y, W, b, _cost_fn, _el_params)
    else:
        print("No Regularisation")
        pred, cost = calc_error(X, y, W, b, _cost_fn, _el_params)

    optimizer = tf.train.GradientDescentOptimizer(_rate).minimize(cost)
    XyWb = [X, y, W, b]
    with tf.Session() as sess:
        if _cross_val == True:
            return hp.cross_validation(sess, XyWb, _train_X, _train_y, _test_X, _test_y, optimizer, cost, huber_cost, _epochs, "lin")
        else:
            return hp.run(sess, XyWb, _train_X, _train_y, _test_X, _test_y, optimizer, cost, huber_cost, _epochs, "lin")

def linear_regression_params(_train_X, _train_y, _test_X, _test_y, _epochs, _rate_list, _cost_fn_list, _regularisation_list):
    X = tf.placeholder(tf.float32, [None, hp.num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")

    W = tf.Variable(tf.random_normal([hp.num_features(_train_X), 1], dtype=tf.float32), name="weight")
    b = tf.Variable(tf.random_normal([1], dtype=tf.float32), name="bias")

    if len(_rate_list) != len(_cost_fn_list) and len(_rate_list) != len(_regularisation_list):
        print("Lists provided are no the same length for cross validation.")
        sys.exit()

    _, huber_cost = huber_error(X, y, W, b)

    _, lad = calc_error_L1(X, y, W, b)

    num_fold = len(_cost_fn_list)

    cost_list = []
    optimizer_list = []
    for i in range(num_fold):
        reg_type, reg_scale = _regularisation_list[i]

        if reg_type == 1:
            pred, cost = calc_error_reg_L1(X, y, W, b, _cost_fn_list[i], reg_scale)
        elif reg_type == 2:
            pred, cost = calc_error_reg_L2(X, y, W, b, _cost_fn_list[i], reg_scale)
        else:
            print("No Regularisation")
            pred, cost = calc_error(X, y, W, b, _cost_fn_list[i])

        cost_list.append(cost)

        optimizer = tf.train.GradientDescentOptimizer(_rate_list[i]).minimize(cost)
        optimizer_list.append(optimizer)

    XyWb = [X, y, W, b]
    with tf.Session() as sess:
        return hp.cv(sess, XyWb, _train_X, _train_y, _test_X, _test_y, optimizer_list, cost_list, _epochs, "lin")
        