import helpers as hp
import linear_regression as linr
import logistic_regression as logr
import neural_network as nn
import support_vector_machine as svm
import sys
import numpy as np
import random

import matplotlib.pyplot as plt

learning_rate_list = [0.1, 0.001, 0.0001, 0.00001]
epochs_list = [100, 500, 1000, 5000, 10000, 50000]

def setup():
    i = input("How many epochs? ")
    j = input("What learning rate? ")
    title = "Basic Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j)
    return i, j, title

def plotter(x, y, tx, ty, filename, title, save):
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

def main():
    print("###############################")
    print("# Machine Learning Coursework #")
    print("######## Douglas Brion ########")
    print("###############################\n")

    ds = hp.load_ds(hp.PATH, hp.FIXED)
    X, y = hp.split(ds)
    # random.shuffle(X)
    # random.shuffle(y)
    # train_X = X
    # train_y = y
    y_size = len(y)
    training_size = 3 * (y_size // 4)
    testing_size = y_size - training_size
    train_X = X[:training_size]
    train_y = y[:training_size]
    test_X = X[-testing_size:]
    test_y = y[-testing_size:]
    # print(train_X)
    # print(train_y)


    if len(sys.argv) == 2:
        if sys.argv[1] == "testsuite":
            print("Running Test Suite")

        elif sys.argv[1] == "plotter":
            print("Running Plotter")
            for i in epochs_list:
                for j in learning_rate_list:
                    print("Starting Linear Regression " + str(i) + " " + str(j))
                    x, y, tx, ty = linr.linear_regression(train_X, train_y, test_X, test_y, i, j)
                    filename = "../figs/TEST_LINR_" + str(i) + "_" + str(j) + ".png"
                    title = "Basic Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j)
                    plotter(x, y, tx, ty, filename, title, True)
                    print("Finished Linear Regression " + str(i) + " " + str(j))

                    print("Starting Logistic Regression " + str(i) + " " + str(j))
                    x, y, tx, ty = logr.logistic_regression(train_X, train_y, test_X, test_y, i, j)
                    filename = "../figs/TEST_LOGR_" + str(i) + "_" + str(j) + ".png"
                    title = "Basic Logistic Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j)
                    plotter(x, y, tx, ty, filename, title, True)
                    print("Finished Logistic Regression " + str(i) + " " + str(j))

                    print("Starting Neural Network " + str(i) + " " + str(j))
                    x, y, tx, ty = nn.neural_network(train_X, train_y, test_X, test_y, i, j)
                    filename = "../figs/TEST_NN_" + str(i) + "_" + str(j) + ".png"
                    title = "Neural Network, Epochs=" + str(i) + ", Learning Rate=" + str(j) + "\nHidden ReLu layer, Output tanh layer"
                    plotter(x, y, tx, ty, filename, title, True)
                    print("Finished Neural Network " + str(i) + " " + str(j))

                    print("\nPlotted epochs " + str(i) + " and learning rate " + str(j) + "\n")
        
        elif sys.argv[1] == "linr":
            i = input("How many epochs? ")
            j = input("What learning rate? ")
            title = "Basic Linear Regression, Epochs=" + i + ", Learning Rate=" + j
            print("Running Linear Regression")
            x, y, tx, ty = linr.linear_regression(train_X, train_y, test_X, test_y, int(i), float(j))
            plotter(x, y, tx, ty, "", title, False)

        elif sys.argv[1] == "linr_break":
            i = 100
            j = 0.31
            incr = 0.01
            for n in range(0,6):
                x, y, tx, ty = linr.linear_regression(train_X, train_y, test_X, test_y, i, j)
                learning_rate = "Learning Rate=" + "{:.2f}".format(j)
                plt.plot(x, y, label=learning_rate)
                plt.title("Basic Linear Regression, Epochs=100, Learning Rate=variable")
                plt.xlabel("Number of Epochs")
                plt.ylabel("Training Error")
                plt.ylim(0, 100)
                plt.legend()
                plt.grid(linestyle='-')
                j += incr
                print(n)
            plt.show()

        elif sys.argv[1] == "logr":
            i = input("How many epochs? ")
            j = input("What learning rate? ")
            title = "Basic Logistic Regression, Epochs=" + i + ", Learning Rate=" + j
            print("Running Logistic Regression")
            x, y, tx, ty = logr.logistic_regression(train_X, train_y, test_X, test_y, int(i), float(j))
            plotter(x, y, tx, ty, "", title, False)

        elif sys.argv[1] == "logr_break":
            i = 100
            j = 0.31
            incr = 0.01
            for n in range(0,6):
                x, y, tx, ty = logr.logistic_regression(train_X, train_y, test_X, test_y, i, j)
                learning_rate = "Learning Rate=" + "{:.2f}".format(j)
                plt.plot(x, y, label=learning_rate)
                plt.title("Basic Logistic Regression, Epochs=100, Learning Rate=variable")
                plt.xlabel("Number of Epochs")
                plt.ylabel("Training Error")
                plt.ylim(0, 100)
                plt.legend()
                plt.grid(linestyle='-')
                j += incr
                print(n)
            plt.show()

        elif sys.argv[1] == "nn":
            i = input("How many epochs? ")
            j = input("What learning rate? ")
            title = "Neural Network, Epochs=" + i + ", Learning Rate=" + j + "\nHidden ReLu layer, Output tanh layer"
            print("Running Neural Network")
            x, y, tx, ty = nn.neural_network(train_X, train_y, test_X, test_y, int(i), float(j))
            plotter(x, y, tx, ty, "", title, False)

        elif sys.argv[1] == "svm":
            i = input("How many epochs? ")
            j = input("What learning rate? ")
            print("Running Support Vector Machine")
            x, y, tx, ty = svm.support_vector_machine(train_X, train_y, test_X, test_y, int(i), float(j))
        else:
            print("Argument was not valid")
    else:
        print("standard")

if __name__ == '__main__':
    main()