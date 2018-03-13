import tensorflow as tf
import helpers as hp
import numpy as np

def softmax_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.softmax(tf.add(tf.matmul(_X, W), b))
    return y, W, b

def selu_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.selu(tf.add(tf.matmul(_X, W), b))
    return y, W, b

def relu_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.relu(tf.add(tf.matmul(_X, W), b))
    return y, W, b

def sigmoid_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.sigmoid(tf.add(tf.matmul(_X, W), b))
    return y, W, b

def tanh_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.tanh(tf.add(tf.matmul(_X, W), b))
    return y, W, b

def cost_function(_y, _pred):
    cost = tf.reduce_mean(tf.square(_y - _pred))
    return cost

def layers(_X, _y):
    inputs = int(hp.num_features(_X))
    hidden_layer_nodes = int((inputs + 1) / 2)


    hidden_layer, hidden_weight, hidden_bias = relu_fn(_X, inputs, hidden_layer_nodes)

    pred, weight, bias = tanh_fn(hidden_layer, hidden_layer_nodes, 1)

    cost = cost_function(_y, pred)
    W = [hidden_weight, weight]
    b = [hidden_bias, bias]

    return pred, cost, W, b

def neural_network(_train_X, _train_y, _epochs, _rate):
    X = tf.placeholder(tf.float32, [None, hp.num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")
    pred, cost, W, b = layers(X, y)
    optimizer = tf.train.GradientDescentOptimizer(_rate).minimize(cost)

    XyWb = [X, y, W, b]

    with tf.Session() as sess:
        return hp.run(sess, XyWb, _train_X, _train_y, optimizer, cost, _epochs, _rate, "nn")

def run_neural_network(epochs, rate):
    ds = hp.load_ds(hp.PATH, hp.FIXED)
    X, y = hp.split(ds)
    return neural_network(X, y, epochs, rate)