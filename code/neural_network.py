import tensorflow as tf
import helpers as hp
import numpy as np

def softmax_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.softmax(tf.add(tf.matmul(_X, W), b))
    return y, W

def selu_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.selu(tf.add(tf.matmul(_X, W), b))
    return y, W

def relu_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.relu(tf.add(tf.matmul(_X, W), b))
    return y, W

def sigmoid_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.sigmoid(tf.add(tf.matmul(_X, W), b))
    return y, W

def tanh_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.nn.tanh(tf.add(tf.matmul(_X, W), b))
    return y, W

def cost_function(_y, _pred):
    cost = tf.reduce_mean(tf.square(_y - _pred))
    return cost

def layers(_X, _y):
    inputs = int(hp.num_features(_X))
    hidden_layer_nodes = int((inputs + 1) / 2)


    hidden_layer, hidden_weight = relu_fn(_X, inputs, hidden_layer_nodes)

    pred, weight = tanh_fn(hidden_layer, hidden_layer_nodes, 1)

    cost = cost_function(_y, pred)
    W = [hidden_weight, weight]

    return pred, cost, W

def neural_network(_train_X, _train_y, _epochs, _rate):
    X = tf.placeholder(tf.float32, [None, hp.num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")
    pred, cost, W = layers(X, y)

    optimizer = tf.train.GradientDescentOptimizer(_rate).minimize(cost)

    init = tf.global_variables_initializer()
    merged_summaries = tf.summary.merge_all()
    x_axis, y_axis = [], []

    with tf.Session() as sess:

        log_directory = 'tmp/logs'
        summary_writer = tf.summary.FileWriter(log_directory, sess.graph)
        sess.run(init)

        for epoch in range(_epochs):
            _, c = sess.run([optimizer, cost], feed_dict={X: _train_X, y: _train_y})

            if (epoch % 10) == 0:
                x_axis.append(epoch+1)
                y_axis.append(c)

            # if (epoch + 1) % 50 == 0:
                #print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
            

        #print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: _train_X, y: _train_y})
        #print("Training cost=", training_cost)
        return x_axis, y_axis

def run_neural_network(epochs, rate):
    ds = hp.load_ds(hp.PATH, hp.FIXED)
    X, y = hp.split(ds)
    return neural_network(X, y, epochs, rate)