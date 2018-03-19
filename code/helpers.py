import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
# import tensorflow.contrib.eager as tfe

PATH = "../data/"
WHITE = "winequality-white.csv"
RED = "winequality-red.csv"
FIXED = "winequality-fixed.csv"
CATEGORIES = "winequality-fixed-categories.csv"

THRESHOLD = 5
LEARNING_RATE = 0.00001




# tfe.enable_eager_execution()

# def huber_loss(_X, _y, _W, _b, m=1.0):
#     pred = _X * _W + _b
#     t = _y - pred
#     return t ** 2 if tf.abs(t) <= m else m * (2 * tf.abs(t) - m)

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
    random.shuffle(data)
    split = np.split(data, [11], axis=1)
    return split[0].astype(np.float32), split[1].astype(np.float32)

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
    
    elif _type == "nn":
        cost = np.sqrt(cost)

            
    
    # print("Cost=", cost, "W=", _sess.run(W), "b=", _sess.run(b), '\n')
    print("Cost=", cost)
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

def data_split_n(_X, _y, _n):
    x_size = len(_X)
    y_size = len(_y)
    delta_size = (y_size % _n)
    new_size = y_size - delta_size
    _X = _X[:-delta_size]
    _y = _y[:-delta_size]
    if x_size != y_size:
        print("Error, X and y not same length")
    else:
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

def run(_sess, _XyWb, _train_X, _train_y, _test_X, _test_y, _opt, _cost, _test_cost, _epochs, _rate, _type):
    X, y, W, b = expand(_XyWb)

    merged_summaries = tf.summary.merge_all()
    log_directory = 'tmp/logs'
    summary_writer = tf.summary.FileWriter(log_directory, _sess.graph)

    training_x_axis, training_y_axis = [], []

    test_x_axis, test_y_axis = [], []
    
    init = tf.global_variables_initializer()
    _sess.run(init)

    for epoch in range(_epochs):
        _sess.run(_opt, feed_dict={X: _train_X, y: _train_y})
        training_cost = _sess.run(_cost, feed_dict={X: _train_X, y: _train_y})
        test_cost = _sess.run(_cost, feed_dict={X: _test_X, y: _test_y})

        if _type == "log":
            training_cost = np.exp(training_cost)
            test_cost = np.exp(test_cost)
        
        elif _type == "nn":
            training_cost = np.sqrt(training_cost)
            test_cost = np.sqrt(test_cost)
            
        if (epoch % 10) == 0:
            training_x_axis.append(epoch + 1)
            training_y_axis.append(training_cost)
            test_x_axis.append(epoch + 1)
            test_y_axis.append(test_cost)

        # if (epoch + 1) % 50 == 0:
            # print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(training_cost), "W=", _sess.run(W), "b=", _sess.run(b))
        
    print("\nOptimization Finished!")
    XyWb = [X, y, W, b]
    print("Training")
    test(_sess, XyWb, _test_cost, _train_X, _train_y, _type)
    print("Testing")
    test(_sess, XyWb, _test_cost, _test_X, _test_y, _type)
    # slope = _sess.run(W)
    # print(slope)
    # y_intercept = _sess.run(b)
    # print(y_intercept)

    # # Get best fit line
    # best_fit = []
    # for i in _train_X:
    #     best_fit.append(tf.matmul(slope, [i + y_intercept]))
    
    # Plot the result
    # print("W: ", _sess.run(W))
    # print("b: ", _sess.run(b))
    # print(_test_X.shape)
    # print(W.shape)
    # best_fit = tf.matmul(_test_X, W)
    # print(best_fit.shape)
    # plt.plot(_test_X, _test_y, 'bo', label='Testing data')
    # plt.plot(_test_X, np.dot(_train_X, _sess.run(W)) + _sess.run(b), label='Fitted line')
    # # plt.legend()
    # plt.show()
    # print(end_cost)
    return training_x_axis, training_y_axis, test_x_axis, test_y_axis    
    # return _train_X, _train_y, _train_X, best_fit


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

        _sess.run(init)
        
        
            
        for i in range(_num_fold):       
            training_cost_sum = 0
            testing_cost_sum = 0
            
            if i == 0:
                train_X = split_X[1]
                train_y = split_y[1]
            else:
                train_X = split_X[0]
                train_y = split_y[0]

            for j in range(_num_fold):
                if j != i:
                    if j > 1:
                        train_X = np.append(train_X, split_X[j], axis=0)
                        train_y = np.append(train_y, split_y[j], axis=0)

            for epoch in range(_epochs):

                _, training_cost = _sess.run([_opt, _cost], feed_dict={X: train_X, y: train_y})
                testing_cost = _sess.run(_cost, feed_dict={X: split_X[i], y: split_y[i]})

                if _type == "log":
                    training_cost = np.exp(training_cost)
                    testing_cost = np.exp(testing_cost)

                if (epoch % 10) == 0:
                    training_x_axis.append((epoch + 1) + i * _epochs)
                    training_y_axis.append(training_cost)
                    testing_x_axis.append((epoch + 1) + i * _epochs)
                    testing_y_axis.append(testing_cost)

                training_cost_sum += training_cost
                testing_cost_sum += testing_cost

            training_cost_sum /= (_num_fold - 1)
            testing_cost_sum /= (_num_fold - 1)

            

                # if (epoch + 1) % 50 == 0:
                    # print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(training_cost), "W=", _sess.run(W), "b=", _sess.run(b))

        print("\nOptimization Finished!\n")

            # XyWb = [X, y, W, b]
        # cost = test(_sess, XyWb, _cost, split_X[i], split_y[i], _type)
            # overall_cost += cost
            # print(overall_cost)
            # training_x_axis.append(i)
            # training_y_axis.append(cost)
            
        # overall_cost /= _num_fold
        # print(overall_cost)
        # training_x_axis.append(_num_fold)
        # training_y_axis.append(overall_cost)
        return training_x_axis, training_y_axis, testing_x_axis, testing_y_axis

def plotter(title, x, y, tx, ty, filename = "", save = False):
    plt.plot(x, y, label='training', color='grey')
    plt.plot(tx, ty, label='test', color='black')
    plt.title(title)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Error (L1 loss)")
    plt.grid(linestyle='-')
    plt.legend()
    if save == True:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
    
def questions():
    i = input("How many epochs? ")
    j = input("What learning rate? ")
    reg_level = input("What regularisation: 0, 1 or 2? ")
    scale = input("What scale for regularisation? ")
    return i, j, reg_level, scale