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

def main():
    hp.intro()
    train_percent = int(input("Enter percentage of data for training: "))
    while train_percent < 0 or train_percent > 100:
        train_percent = int(input("Error. Value not a valid percentage. Enter a new percentage: "))

    print(train_percent, "% of the data will be used for training")
    test_percent = 100 - train_percent
    print(test_percent, "% of the data will be used for testing")
    
    ds = hp.load_ds(hp.PATH, hp.FIXED)
    X, y = hp.split(ds)

    y_size = len(y)
    training_size = int((train_percent / 100) * y_size)
    testing_size = y_size - training_size

    train_X, train_y, test_X, test_y = hp.random_train_test(X, y, training_size)

    if len(sys.argv) == 2:
        if sys.argv[1] == "testsuite":
            print("Running Test Suite")

        elif sys.argv[1] == "hp.plotter":
            print("Running Plotter")
            for i in epochs_list:
                for j in learning_rate_list:
                    print("Starting Linear Regression " + str(i) + " " + str(j))
                    x, y, tx, ty = linr.linear_regression(train_X, train_y, test_X, test_y, i, j)
                    filename = "../figs/TEST_LINR_" + str(i) + "_" + str(j) + ".png"
                    title = "Basic Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j)
                    hp.plotter(title, x, y, tx, ty, filename, True)
                    print("Finished Linear Regression " + str(i) + " " + str(j))

                    print("Starting Logistic Regression " + str(i) + " " + str(j))
                    x, y, tx, ty = logr.logistic_regression(train_X, train_y, test_X, test_y, i, j)
                    filename = "../figs/TEST_LOGR_" + str(i) + "_" + str(j) + ".png"
                    title = "Basic Logistic Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j)
                    hp.plotter(title, x, y, tx, ty, filename, True)
                    print("Finished Logistic Regression " + str(i) + " " + str(j))

                    print("Starting Neural Network " + str(i) + " " + str(j))
                    x, y, tx, ty = nn.neural_network(train_X, train_y, test_X, test_y, i, j)
                    filename = "../figs/TEST_NN_" + str(i) + "_" + str(j) + ".png"
                    title = "Neural Network, Epochs=" + str(i) + ", Learning Rate=" + str(j) + "\nHidden ReLu layer, Output tanh layer"
                    hp.plotter(title, x, y, tx, ty, filename, True)
                    print("Finished Neural Network " + str(i) + " " + str(j))

                    print("\nPlotted epochs " + str(i) + " and learning rate " + str(j) + "\n")
        
        elif sys.argv[1] == "linr":
            i = input("How many epochs? ")
            j = input("What learning rate? ")
            title = "Basic Linear Regression, Epochs=" + i + ", Learning Rate=" + j
            print("Running Linear Regression")
            x, y, tx, ty = linr.linear_regression(train_X, train_y, test_X, test_y, int(i), float(j), 0, True)
            hp.plotter(title, x, y, tx, ty)

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
            hp.plotter(title, x, y, tx, ty)

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
            x, y, tx, ty = nn.neural_network(train_X, train_y, test_X, test_y, int(i), float(j), 0, True)
            hp.plotter(title, x, y, tx, ty)

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