import tensorflow as tf
import numpy as np
import helpers as hp

def calc_error(_X, _y, _W, _b):
    pred = tf.add(tf.matmul(_X, _W), _b)
    # cost = tf.reduce_mean(tf.reduce_sum(tf.matmul(tf.transpose(_y), tf.log(pred)), reduction_indices=1))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=_y))
    return pred, cost

def logistic_regression(_train_X, _train_y, _epochs, _rate):
    X = tf.placeholder(tf.float32, [None, hp.num_features(_train_X)], name="input")
    y = tf.placeholder(tf.float32, name="output")

    W = tf.Variable(tf.zeros([hp.num_features(_train_X), 1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    # W = tf.Variable(tf.random_normal([hp.num_features(_train_X), 1], dtype=tf.float32), name="weight")
    # b = tf.Variable(tf.random_normal([1], dtype=tf.float32), name="bias")

    _, cost = calc_error(X, y, W, b)

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
            
            c = np.exp(c)

            if (epoch % 10) == 0:
                x_axis.append(epoch + 1)
                y_axis.append(c)

            if (epoch + 1) % 50 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", c)
            
        #print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: _train_X, y: _train_y})
        #print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
        return x_axis, y_axis

def run_logistic_regression(epochs, rate):
    ds = hp.load_ds(hp.PATH, hp.FIXED)
    X, y = hp.split(ds)
    return logistic_regression(X, y, epochs, rate)