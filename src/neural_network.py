import tensorflow as tf
import helpers as hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.0001
epochs = 100000
step = 50

def tanh_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.tanh(tf.matmul(_X, W) + b)
    return y, W

def cost_function(_y, _pred):
    cost = tf.reduce_mean(tf.square(_y - _pred))   
    return cost

def neural_net(_X, _y):
    inputs = int(hp.num_features(_X))
    hidden_layer_nodes = int((inputs + 1) / 2)


    hidden_layer, hidden_weight = tanh_fn(_X, inputs, hidden_layer_nodes)

    pred, weight = tanh_fn(hidden_layer, hidden_layer_nodes, 1)

    cost = cost_function(_y, pred)
    W = [hidden_weight, weight]
    print(W)


    return pred, cost, W

def run_neural_net(_train_X, _train_y):
    X = tf.placeholder(tf.float32, [None, hp.num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")
    pred, cost, W = neural_net(X, y)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    merged_summaries = tf.summary.merge_all()
    plot_point = [[], []]

    with tf.Session() as sess:

        log_directory = 'tmp/logs'
        summary_writer = tf.summary.FileWriter(log_directory, sess.graph)
        sess.run(init)

        for epoch in range(epochs):
            _, c = sess.run([optimizer, cost], feed_dict={X: _train_X, y: _train_y})

            if (epoch % 10) == 0:
                plot_point[0].append(epoch+1)
                plot_point[1].append(c)

            if (epoch + 1) % step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
            

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: _train_X, y: _train_y})
        print("Training cost=", training_cost)
        # plt.plot(_train_X, _train_y, 'ro', label='Original Data', marker='.')
        # # plt.plot(_train_X, np.dot(_train_X, sess.run(W)) + sess.run(b), label='Fitted line')
        # plt.legend()
        # plt.show()

        plt.plot(plot_point[0], plot_point[1])
        plt.xlabel('Number of Epochs')
        plt.ylabel('Training Error')
        plt.grid(linestyle='-')
        plt.show()

ds = hp.load_ds(hp.PATH, hp.FIXED)
X, y = hp.split(ds)
run_neural_net(X, y)