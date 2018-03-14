import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

PATH = "../data/"
WHITE = "winequality-white.csv"
RED = "winequality-red.csv"
FIXED = "winequality-fixed.csv"
CATEGORIES = "winequality-fixed-categories.csv"

THRESHOLD = 5
LEARNING_RATE = 0.00001

def intro():
    print("###############################")
    print("# Machine Learning Coursework #")
    print("######## Douglas Brion ########")
    print("###############################\n")

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

def remove_elements(a, b):
    return np.setxor1d(a, b)

def expand(_lst):
    if len(_lst) != 4:
        print("Error, list not 4 long")
    else:
        return _lst[0], _lst[1], _lst[2], _lst[3]

def test(_sess, _XyWb, _cost, _test_X, _test_y, _type):
    X, y, W, b = expand(_XyWb)
    cost = _sess.run(_cost, feed_dict={X: _test_X, y: _test_y})

    if _type == "log":
        cost = np.exp(cost)
    
    print("Cost=", cost, "W=", _sess.run(W), "b=", _sess.run(b), '\n')
    return cost

def random_train_test(_X, _y, _train_size):
    y_size = len(_y)
    x_size = len(_X)
    if y_size != x_size:
        print("Something has gone wrong, arrays not same length")
        print("length y: ", y_size)
        print("length x: ", x_size)
    else:
        indexes = np.arange(y_size)
        train_indexes = np.random.choice(y_size, _train_size, replace=False)
        test_indexes = remove_elements(indexes, train_indexes)
        train_X_sample = _X[train_indexes]
        train_y_sample = _y[train_indexes]
        test_X_sample = _X[test_indexes]
        test_y_sample = _y[test_indexes]
        return train_X_sample, train_y_sample, test_X_sample, test_y_sample

# 10 fold cross validaiton split
def cross_validation_split(data):
    print("Splitting Data...")
    n = len(data)
    random.shuffle(data)

    output = np.array_split(data, p.CV_SPLIT)

    return output

    
def data_split_n(_X, _y, _n):
    x_size = len(_X)
    y_size = len(_y)
    if x_size != y_size:
        print("Error, X and y not same length")
    else:
        random.shuffle(_X)
        random.shuffle(_y)
        
        out_X = np.array_split(_X, _n)
        out_y = np.array_split(_y, _n)
        return out_X, out_y

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

def run(_sess, _XyWb, _train_X, _train_y, _test_X, _test_y, _opt, _cost, _epochs, _rate, _type):
    X, y, W, b = expand(_XyWb)

    merged_summaries = tf.summary.merge_all()
    log_directory = 'tmp/logs'
    summary_writer = tf.summary.FileWriter(log_directory, _sess.graph)

    training_x_axis, training_y_axis = [], []

    test_x_axis, test_y_axis = [], []
    
    init = tf.global_variables_initializer()
    _sess.run(init)

    for epoch in range(_epochs):
        _, training_cost = _sess.run([_opt, _cost], feed_dict={X: _train_X, y: _train_y})
        test_cost = _sess.run(_cost, feed_dict={X: _test_X, y: _test_y})

        if _type == "log":
            training_cost = np.exp(training_cost)
            test_cost = np.exp(test_cost)
            
        if (epoch % 10) == 0:
            training_x_axis.append(epoch + 1)
            training_y_axis.append(training_cost)
            test_x_axis.append(epoch + 1)
            test_y_axis.append(test_cost)

        # if (epoch + 1) % 50 == 0:
            # print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(training_cost), "W=", _sess.run(W), "b=", _sess.run(b))
        
    print("\nOptimization Finished!")
    XyWb = [X, y, W, b]
    test(_sess, XyWb, _cost, _train_X, _train_y, _type)
    test(_sess, XyWb, _cost, _test_X, _test_y, _type)
    # print(end_cost)
    return training_x_axis, training_y_axis, test_x_axis, test_y_axis

def cross_validation(_sess, _XyWb, _train_X, _train_y, _test_X, _test_y, _opt, _cost, _epochs, _rate,  _type, _num_fold=10):
    X, y, W, b = expand(_XyWb)

    merged_summaries = tf.summary.merge_all()
    log_directory = 'tmp/logs'
    summary_writer = tf.summary.FileWriter(log_directory, _sess.graph)

    training_x_axis, training_y_axis = [], []
    testing_x_axis, testing_y_axis = [], []
    
    init = tf.global_variables_initializer()

    y_size = len(_train_y)
    x_size = len(_train_X)

    if y_size != x_size:
        print("Something has gone wrong, arrays not same length")
        print("length y: ", y_size)
        print("length x: ", x_size)

    else:
        overall_cost = 0
        split_X, split_y = data_split_n(_train_X, _train_y, _num_fold)

        for i in range(_num_fold):       
            _sess.run(init)
            for epoch in range(_epochs):
                cost_sum = 0
                blah = 0
                for j in range(_num_fold):

                    if j == i:
                        continue

                    train_X = split_X[j]
                    train_y = split_y[j]

                    _, training_cost = _sess.run([_opt, _cost], feed_dict={X: train_X, y: train_y})
                    test_cost = _sess.run(_cost, feed_dict={X: _test_X, y: _test_y})

                    if _type == "log":
                        training_cost = np.exp(training_cost)
                        test_cost = np.exp(test_cost)

        
                    cost_sum += training_cost
                    blah += test_cost


                cost_sum /= (_num_fold - 1)
                blah /= (_num_fold - 1)

                if (epoch % 10) == 0:
                    training_x_axis.append(epoch + 1)
                    training_y_axis.append(cost_sum)
                    testing_x_axis.append(epoch + 1)
                    testing_y_axis.append(blah)

                # if (epoch + 1) % 50 == 0:
                    # print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(training_cost), "W=", _sess.run(W), "b=", _sess.run(b))

            print("\nOptimization Finished!\n")

            XyWb = [X, y, W, b]
            cost = test(_sess, XyWb, _cost, split_X[i], split_y[i], _type)
            overall_cost += cost

            # training_x_axis.append(i)
            # training_y_axis.append(cost)
            
        overall_cost /= _num_fold
        print(overall_cost)
        # training_x_axis.append(_num_fold)
        # training_y_axis.append(overall_cost)
        return training_x_axis, training_y_axis, testing_x_axis, testing_y_axis

def plotter(title, x, y, tx, ty, filename = "", save = False):
    plt.plot(x, y, label='training')
    plt.plot(tx, ty, label='test')
    plt.title(title)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Error")
    plt.grid(linestyle='-')
    plt.legend()
    if save == True:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()