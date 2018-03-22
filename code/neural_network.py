import tensorflow as tf
import helpers as hp
import numpy as np

def softmax_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.softmax(tf.matmul(_X, W) + b)
    return y, W, b

def selu_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.selu(tf.add(tf.matmul(_X, W) , b))
    return y, W, b

def relu_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.relu(tf.add(tf.matmul(_X, W) , b))
    return y, W, b

def sigmoid_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.sigmoid(tf.add(tf.matmul(_X, W) , b))
    return y, W, b

def tanh_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.tanh(tf.add(tf.matmul(_X, W) , b))
    return y, W, b

def calc_error_L1(_y, _pred):
    print("Loss Function L1")
    cost = tf.reduce_mean(tf.abs(_y - _pred))
    return cost

def huber_error(_y, _pred, _delta=1.0):
    residual = tf.abs(_y - _pred)
    cond = tf.less(residual, _delta)
    small_res = 0.5 * tf.square(residual)
    large_res = _delta * residual - 0.5 * tf.square(_delta)
    cost = tf.reduce_mean(tf.where(cond, small_res, large_res))
    return cost

def cost_function(_y, _pred):
    cost = tf.reduce_mean(tf.square(_y - _pred))
    return cost

def layers(_X, _y, _output_layer=0):
    inputs = int(hp.num_features(_X))
    hidden_layer_nodes = int((inputs + 1) / 2)

    hidden_layer, hidden_weight, hidden_bias = relu_fn(_X, inputs, hidden_layer_nodes)

    if _output_layer == 0:
        print("Ouput Layer is ReLU")
        pred, weight, bias = relu_fn(hidden_layer, hidden_layer_nodes, 1)
    elif _output_layer == 1: 
        print("Ouput Layer is SeLU")
        pred, weight, bias = selu_fn(hidden_layer, hidden_layer_nodes, 1)
    elif _output_layer == 2:
        print("Ouput Layer is Softmax")
        pred, weight, bias = softmax_fn(hidden_layer, hidden_layer_nodes, 1)
    elif _output_layer == 3:
        print("Ouput Layer is TanH")
        pred, weight, bias = tanh_fn(hidden_layer, hidden_layer_nodes, 1)
    elif _output_layer == 4:
        print("Ouput Layer is Sigmoid")
        pred, weight, bias = sigmoid_fn(hidden_layer, hidden_layer_nodes, 1)
    else:
        print("Ouput Layer is ReLU")
        pred, weight, bias = relu_fn(hidden_layer, hidden_layer_nodes, 1)

    cost = cost_function(_y, pred)
    W = [hidden_weight, weight]
    b = [hidden_bias, bias]

    return pred, cost, W, b

def neural_network(_train_X, _train_y, _test_X, _test_y, _epochs, _rate,  _regularisation, _cross_val, _output_layer=0):
    reg_type, reg_scale = _regularisation
    X = tf.placeholder(tf.float32, [None, hp.num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")
    pred, cost, W, b = layers(X, y, _output_layer)

    lad = calc_error_L1(y, pred)
    huber_loss = huber_error(y, pred)

    print("Regularisation: ", _regularisation)
    if reg_type == 1:
        L1 = tf.contrib.layers.l1_regularizer(scale=reg_scale)
        reg_cost = tf.contrib.layers.apply_regularization(L1, W)
    elif reg_type == 2:
        L2 = tf.contrib.layers.l2_regularizer(scale=reg_scale)
        reg_cost = tf.contrib.layers.apply_regularization(L2, W)
    else:
        reg_cost = 0
    
    cost += reg_cost

    optimizer = tf.train.GradientDescentOptimizer(_rate).minimize(cost)

    XyWb = [X, y, W, b]

    with tf.Session() as sess:
        if _cross_val == True:
            return hp.cross_validation(sess, XyWb, _train_X, _train_y, _test_X, _test_y, optimizer, cost, huber_loss, _epochs, "nn")
        else:
            return hp.run(sess, XyWb, _train_X, _train_y, _test_X, _test_y, optimizer, cost, huber_loss, _epochs, "nn")
