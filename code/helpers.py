import pandas as pd
import numpy as np
import tensorflow as tf
import random

PATH = "../data/"
WHITE = "winequality-white.csv"
RED = "winequality-red.csv"
FIXED = "winequality-fixed.csv"
CATEGORIES = "winequality-fixed-categories.csv"

THRESHOLD = 5
LEARNING_RATE = 0.00001

def num_examples(_ds):
    return _ds.shape[0]

def num_features(_ds):
    return _ds.shape[1]

def load_ds(_path, _infile):
    ds = pd.read_csv(_path + _infile, sep = ',')
    return ds

def split(_ds):
    data = _ds.values
    split = np.split(data, [11], axis=1)
    return split[0], split[1]

def expand(_lst):
    if len(_lst) != 4:
        print("Error, list not 4 long")
    else:
        return _lst[0], _lst[1], _lst[2], _lst[3]

def test(_sess, _XyWb, _cost, _train_X, _train_y, _type):
    X, y, W, b = expand(_XyWb)
    cost = _sess.run(_cost, feed_dict={X: _train_X, y: _train_y})

    if _type == "log":
        cost = np.exp(cost)
    
    print("Cost=", cost, "W=", _sess.run(W), "b=", _sess.run(b), '\n')
    return cost

def random_sample(_X, _y, _size):
    y_size = len(_y)
    x_size = len(_X)
    if y_size != x_size:
        print("Something has gone wrong, arrays not same length")
        print("length y: ", y_size)
        print("length x: ", x_size)
    else:
        index_sample = np.random.choice(y_size, _size, replace=False)
        X_sample = _X[index_sample]
        y_sample = _y[index_sample]
        return X_sample, y_sample

def run(_sess, _XyWb, _train_X, _train_y, _opt, _cost, _epochs, _rate, _type):
    X, y, W, b = expand(_XyWb)

    merged_summaries = tf.summary.merge_all()
    log_directory = 'tmp/logs'
    summary_writer = tf.summary.FileWriter(log_directory, _sess.graph)

    x_axis, y_axis = [], []
    
    init = tf.global_variables_initializer()
    _sess.run(init)

    for epoch in range(_epochs):
        _, c = _sess.run([_opt, _cost], feed_dict={X: _train_X, y: _train_y})

        if _type == "log":
            c = np.exp(c)
            
        if (epoch % 10) == 0:
            x_axis.append(epoch + 1)
            y_axis.append(c)

        if (epoch + 1) % 50 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W=", _sess.run(W), "b=", _sess.run(b))
        
    print("\nOptimization Finished!\n")
    XyWb = [X, y, W, b]
    test(_sess, XyWb, _cost, _train_X, _train_y, _type)

    return x_axis, y_axis

def cross_validation(_sess, _XyWb, _train_X, _train_y, _opt, _cost, _epochs, _rate, _num_fold, _type):
    X, y, W, b = expand(_XyWb)

    merged_summaries = tf.summary.merge_all()
    log_directory = 'tmp/logs'
    summary_writer = tf.summary.FileWriter(log_directory, _sess.graph)

    x_axis, y_axis = [], []
    
    init = tf.global_variables_initializer()

    y_size = len(_train_y)
    x_size = len(_train_X)

    if y_size != x_size:
        print("Something has gone wrong, arrays not same length")
        print("length y: ", y_size)
        print("length x: ", x_size)

    else:
        sample_size = y_size // _num_fold
        overall_cost = 0

        for i in range(_num_fold):       
            _sess.run(init)
            for epoch in range(_epochs):
                cost_sum = 0
                for j in range(_num_fold):

                    if j == i:
                        continue

                    train_X, train_y = random_sample(_train_X, _train_y, sample_size)

                    _, c = _sess.run([_opt, _cost], feed_dict={X: train_X, y: train_y})

                    if _type == "log":
                        c = np.exp(c)

                    cost_sum += c


                cost_sum /= (_num_fold - 1)

            print("\nOptimization Finished!\n")

            XyWb = [X, y, W, b]
            cost = test(_sess, XyWb, _cost, _train_X, _train_y, _type)
            overall_cost += cost

            x_axis.append(i)
            y_axis.append(error_n)
            
        overall_cost /= _num_fold
        x_axis.append(_num_fold)
        y_axis.append(overall_cost)
        return x_axis, y_axis