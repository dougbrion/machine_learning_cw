import tensorflow as tf
import helpers as hp

def sigmoid_fn(_X, _units):
    inputs = _X.get_shape()[1].value
    W = tf.Variable(tf.random_normal([inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.sigmoid(tf.matmul(_X, W) + b)
    return y

def tanh_fn(_X, _units):
    inputs = _X.get_shape()[1].value
    W = tf.Variable(tf.random_normal([inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.tanh(tf.matmul(_X, W) + b)
    return y

def neural_net_regression(_X, _y, _sel):
    hidden_layer_nodes = (hp.num_features(_X) + 1) // 2

    hidden_layer = tanh_fn(_X, hidden_layer_nodes)

    if _sel == "sigmoid":
        prediction, w2 = l_sigmoid(hidden_layer, hl_nodes, 1)
    elif _sel == "tanh":
        prediction, w2 = l_tanh(hidden_layer, hl_nodes, 1)

    error = tf.reduce_mean(tf.square(y - prediction))

    w = [w1, w2]

return [ prediction, error, w ]