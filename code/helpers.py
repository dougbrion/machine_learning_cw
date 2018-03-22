import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import sys
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

def run(_sess, _XyWb, _train_X, _train_y, _test_X, _test_y, _opt, _cost, _test_cost, _epochs, _type):
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
            
        if (epoch % 10) == 0:
            training_x_axis.append(epoch + 1)
            training_y_axis.append(training_cost)
            test_x_axis.append(epoch + 1)
            test_y_axis.append(test_cost)

    print("\nOptimization Finished!")
    XyWb = [X, y, W, b]
    print("Training")
    train_test = test(_sess, XyWb, _test_cost, _train_X, _train_y, _type)
    print("Testing")
    test_test = test(_sess, XyWb, _test_cost, _test_X, _test_y, _type)
    training_y_axis.append(train_test)
    test_y_axis.append(test_test)

    return training_x_axis, training_y_axis, test_x_axis, test_y_axis    
    # return _train_X, _train_y, _train_X, best_fit


def cv(_sess, _XyWb, _train_X, _train_y, _test_X, _test_y, _opt_list, _cost_list, _epochs, _type):
    X, y, W, b = expand(_XyWb)

    y_size = len(_train_y)
    x_size = len(_train_X)

    if len(_cost_list) != len(_opt_list):
        print("Optimiser and Cost lists not that same length")
        print("Length of Optimiser list: ", len(_opt_list))
        print("Length of Cost list: ", len(_cost_list))
        sys.exit()
    
    if y_size != x_size:
        print("Something has gone wrong, arrays not same length")
        print("length y: ", y_size)
        print("length x: ", x_size)
        sys.exit()
    
    num_fold = len(_cost_list)

    split_X, split_y = data_split_n(_train_X, _train_y, num_fold)

    training_x_axis, training_y_axis = [], []
    testing_x_axis, testing_y_axis = [], []

    for i in range(num_fold):       
        training_cost_sum = 0
        testing_cost_sum = 0
        
        if i == 0:
            train_X = split_X[1]
            train_y = split_y[1]
        else:
            train_X = split_X[0]
            train_y = split_y[0]

        for j in range(num_fold):
            if j != i:
                if j > 1:
                    train_X = np.append(train_X, split_X[j], axis=0)
                    train_y = np.append(train_y, split_y[j], axis=0)

        init = tf.global_variables_initializer()

        _sess.run(init)

        for epoch in range(_epochs):
            _, training_cost = _sess.run([_opt_list[i], _cost_list[i]], feed_dict={X: train_X, y: train_y})
            testing_cost = _sess.run(_cost_list[i], feed_dict={X: split_X[i], y: split_y[i]})

            if (epoch % 10) == 0:
                training_x_axis.append((epoch + 1) + i * _epochs)
                training_y_axis.append(training_cost)
                testing_x_axis.append((epoch + 1) + i * _epochs)
                testing_y_axis.append(testing_cost)

            training_cost_sum += training_cost
            testing_cost_sum += testing_cost

        training_cost_sum /= (num_fold - 1)
        testing_cost_sum /= (num_fold - 1)

    print("\nOptimization Finished!\n")

    return training_x_axis, training_y_axis, testing_x_axis, testing_y_axis


def cross_validation(_sess, _XyWb, _train_X, _train_y, _test_X, _test_y, _opt, _cost, _test_cost, _epochs, _type, _num_fold=10):
    X, y, W, b = expand(_XyWb)

    merged_summaries = tf.summary.merge_all()
    log_directory = 'tmp/logs'
    summary_writer = tf.summary.FileWriter(log_directory, _sess.graph)

    training_x_axis, training_y_axis = [[]], [[]]
    testing_x_axis, testing_y_axis = [[]], [[]]
    straining_x_axis, straining_y_axis = [], []
    stesting_x_axis, stesting_y_axis = [], []

    y_size = len(_train_y)
    x_size = len(_train_X)

    for a in range(_num_fold - 1):
        training_x_axis.append([])
        training_y_axis.append([])
        testing_x_axis.append([])
        testing_y_axis.append([])

    if y_size != x_size:
        print("Something has gone wrong, arrays not same length")
        print("length y: ", y_size)
        print("length x: ", x_size)

    else:
        overall_test_cost = 0
        overall_train_cost = 0
        split_X, split_y = data_split_n(_train_X, _train_y, _num_fold)
        
        for i in range(_num_fold):       

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

            init = tf.global_variables_initializer()

            _sess.run(init)

            for epoch in range(_epochs):

                _, training_cost = _sess.run([_opt, _cost], feed_dict={X: train_X, y: train_y})
                testing_cost = _sess.run(_cost, feed_dict={X: split_X[i], y: split_y[i]})
                
                if (epoch % 10) == 0:
                    training_x_axis[i].append(epoch + 1)
                    training_y_axis[i].append(training_cost)
                    testing_x_axis[i].append(epoch + 1)
                    testing_y_axis[i].append(testing_cost)
            
            XyWb = [X, y, W, b]
            train_test = test(_sess, XyWb, _test_cost, train_X, train_y, _type)
            test_test = test(_sess, XyWb, _test_cost, split_X[i], split_y[i], _type)

            overall_test_cost += test_test
            overall_train_cost += train_test

        overall_test_cost /= _num_fold       # if (epoch + 1) % 50 == 0:
        overall_train_cost /= _num_fold       # if (epoch + 1) % 50 == 0:

        tmpa  = 0
        tmpb = 0

        print(len(training_x_axis[0]))

        for i in range(len(training_x_axis[0])):
            for j in range(_num_fold):
                tmpa += (training_y_axis[j][i] / _num_fold)
                tmpb += (testing_y_axis[j][i] / _num_fold)
            straining_y_axis.append(tmpa)
            stesting_y_axis.append(tmpb)
            tmpa = 0
            tmpb = 0

        straining_y_axis.append(overall_train_cost)
        stesting_y_axis.append(overall_test_cost)
        print(len(straining_y_axis))
        print("\nOptimization Finished!\n")
        # return training_x_axis, training_y_axis, testing_x_axis, testing_y_axis

        return training_x_axis[0], straining_y_axis, testing_x_axis[0], stesting_y_axis

def plotter(title, x, y, tx, ty, percent=100, filename = "", save = False):
    train_label = "training: " + str(percent) + "%\nfinal error: " + "{:.3f}".format(y[-2])
    test_label = "testing: " + str(100 - percent) + "%\nfinal error: " + "{:.3f}".format(ty[-2])
    compare_train = "huber train error: " + "{:.3f}".format(y[-1])
    compare_test = "huber test error: " + "{:.3f}".format(ty[-1])
    plt.plot(x, y[:-1], label=train_label, color='grey')
    plt.plot(tx, ty[:-1], label=test_label, color='black')
    plt.plot(x[-1], y[-1], label=compare_train, color='grey', marker='.')
    plt.plot(tx[-1], ty[-1], label=compare_test, color='black', marker='.')
    plt.title(title)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Error")
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

def histogram():
    ds = load_ds(PATH, FIXED)
    data = ds['quality'].values
    y = data.astype(np.int64)
    print(y)
    # list_y = np.concatenate(y)
    # print(list_y)
    count = np.bincount(y)
    print(count)
    count = count[3:]
    print(count)
    val_range = np.arange(10)
    val_range = val_range[3:]
    percent = np.multiply(np.divide(count, len(y)), 100)
    print(val_range)
    print(percent)
    plt.bar(val_range, count, width=0.75, linewidth=0.5, edgecolor="black", color="grey")
    plt.xlabel("Wine Quality Value")
    plt.ylabel("Number of Wine Samples")
    i = 0
    for a, b in zip(val_range, count):
        plt.text(a - 0.5, b + 20, str(b) + " Samples")
        plt.text(a - 0.3, b + 90, "{:.2f}".format(percent[i]) + "%")
        i = i + 1
    # plt.annotate()
    plt.show()

def cross_val():
    if input("Would you like to use cross validation? (y/n)? ") == "y":
        return True
    else: 
        return False

def percentages():
    train_percent = int(input("Enter percentage of data for training: "))
    while train_percent < 0 or train_percent > 100:
        train_percent = int(input("Error. Value not a valid percentage. Enter a new percentage: "))

    print(train_percent, "% of the data will be used for training")
    test_percent = 100 - train_percent
    print(test_percent, "% of the data will be used for testing")
    return train_percent, test_percent

def getXy():
    ds = load_ds(PATH, FIXED)
    X, y = split(ds)
    return X, y

def intro():
    print("###############################")
    print("# Machine Learning Coursework #")
    print("######## Douglas Brion ########")
    print("###############################\n")

    train_percent, test_percent = percentages()
    cross_val_bool = cross_val()
    X, y = getXy()

    training_size = int((train_percent / 100) * len(y))
    testing_size = len(y) - training_size

    train_X, train_y, test_X, test_y = random_train_test(X, y, training_size)
    return train_X, train_y, test_X, test_y, train_percent, cross_val_bool

def cv_intro():
    X, y = getXy()
    training_size = int((100 / 100) * len(y))
    train_X, train_y, test_X, test_y = random_train_test(X, y, training_size)
    return train_X, train_y, test_X, test_y