import helpers as hp
import linear_regression as linr
import logistic_regression as logr
import neural_network as nn
import elastic_net as en
import sys
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe


def main():
        # make results reproducible
    seed = 13
    np.random.seed(seed)
    tf.set_random_seed(seed)
    learning_rate_list = [0.1, 0.001, 0.0001, 0.00001]
    epochs_list = [100, 500, 1000, 5000, 10000, 50000]

    regularisation = 1, 0.01    
    hp.intro()
    train_percent = int(input("Enter percentage of data for training: "))
    while train_percent < 0 or train_percent > 100:
        train_percent = int(input("Error. Value not a valid percentage. Enter a new percentage: "))

    print(train_percent, "% of the data will be used for training")
    test_percent = 100 - train_percent
    print(test_percent, "% of the data will be used for testing")
    
    if input("Would you like to use cross validation? (y/n)? ") == "y":
        cross_val = True
    else: 
        cross_val = False

    
    ds = hp.load_ds(hp.PATH, hp.FIXED)
    X, y = hp.split(ds)

    y_size = len(y)
    training_size = int((train_percent / 100) * y_size)
    testing_size = y_size - training_size

    train_X, train_y, test_X, test_y = hp.random_train_test(X, y, training_size)
    

    if len(sys.argv) == 2:
        if sys.argv[1] == "hp.plotter":
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
            i = int(input("How many epochs? "))
            j = float(input("What learning rate? "))
            cost_fn = int(input("Would you like a 1 (L1), 2 (L2), 3 (Elastic Net), 4 (SVM), 5 (Huber Loss) cost function? "))
            reg_level = int(input("What regularisation: 0, 1 or 2? "))
            scale = float(input("What scale for regularisation? "))
            regularisation = reg_level, scale
            title = "Basic Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j)
            print("Running Linear Regression")
            x, y, tx, ty = linr.linear_regression(train_X, train_y, test_X, test_y, i, j, cost_fn, regularisation, cross_val)
            hp.plotter(title, x, y, tx, ty)

        elif sys.argv[1] == "linr_break":
            i = 100
            j = 0.31
            incr = 0.01
            for n in range(0,6):
                x, y, tx, ty = linr.linear_regression(train_X, train_y, test_X, test_y, i, j, regularisation, cross_val)
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

        elif sys.argv[1] == "elastic":
            i = input("How many epochs? ")
            j = input("What learning rate? ")
            title = "Elastic Net  LinearRegression, Epochs=" + i + ", Learning Rate=" + j
            print("Running Elastic Net Regression")
            x, y, tx, ty = en.elastic_net(train_X, train_y, test_X, test_y, int(i), float(j), regularisation, cross_val)
            hp.plotter(title, x, y, tx, ty)

        # elif sys.argv[1] == "logr":
        #     i = input("How many epochs? ")
        #     j = input("What learning rate? ")
        #     title = "Basic Logistic Regression, Epochs=" + i + ", Learning Rate=" + j
        #     print("Running Logistic Regression")
        #     x, y, tx, ty = logr.logistic_regression(train_X, train_y, test_X, test_y, int(i), float(j), regularisation, cross_val)
        #     hp.plotter(title, x, y, tx, ty)

        # elif sys.argv[1] == "logr_break":
        #     i = 100
        #     j = 0.31
        #     incr = 0.01
        #     for n in range(0,6):
        #         x, y, tx, ty = logr.logistic_regression(train_X, train_y, test_X, test_y, i, j, regularisation, cross_val)
        #         learning_rate = "Learning Rate=" + "{:.2f}".format(j)
        #         plt.plot(x, y, label=learning_rate)
        #         plt.title("Basic Logistic Regression, Epochs=100, Learning Rate=variable")
        #         plt.xlabel("Number of Epochs")
        #         plt.ylabel("Training Error")
        #         plt.ylim(0, 100)
        #         plt.legend()
        #         plt.grid(linestyle='-')
        #         j += incr
        #         print(n)
        #     plt.show()

        elif sys.argv[1] == "histogram":
            ds = hp.load_ds(hp.PATH, hp.FIXED)
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

        elif sys.argv[1] == "nn":
            i,j,reg_level,scale = hp.questions()
            regularisation = reg_level, scale
            title = "Neural Network, Epochs=" + i + ", Learning Rate=" + j + "\nHidden ReLu layer, Output tanh layer"
            print("Running Neural Network")
            x, y, tx, ty = nn.neural_network(train_X, train_y, test_X, test_y, int(i), float(j), regularisation, cross_val)
            hp.plotter(title, x, y, tx, ty)
        
        elif sys.argv[1] == "L1lambda":
            i = input("How many epochs? ")
            j = input("What learning rate? ")
            init_reg = 0.001
            for n in range(30):
                regularisation = 1, init_reg
                x, y, tx, ty = nn.neural_network(train_X, train_y, test_X, test_y, int(i), float(j), regularisation, cross_val)
                reg_l = "Reg Lambda=" + "{:.3f}".format(init_reg)
                plt.plot(x, y, label="train" + reg_l)
                plt.plot(tx, ty, label="test" + reg_l)
                plt.title("Basic Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j))
                plt.xlabel("Number of Epochs")
                plt.ylabel("Training Error")
                # plt.ylim(0, 100)
                plt.legend()
                plt.grid(linestyle='-')
                init_reg += 0.001
            plt.show()


        elif sys.argv[1] == "L2lambda":
            i = input("How many epochs? ")
            j = input("What learning rate? ")
            init_reg = 0.001
            for i in range(7):
                regularisation = 2, init_reg
                x, y, tx, ty = nn.neural_network(train_X, train_y, test_X, test_y, int(i), float(j), regularisation, cross_val)
                reg_l = "Reg Lambda=" + "{:.2f}".format(init_reg)
                plt.plot(x, y, label="train" + reg_l)
                plt.plot(tx, ty, label="test" + reg_l)
                plt.title("Basic Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j))
                plt.xlabel("Number of Epochs")
                plt.ylabel("Training Error")
                # plt.ylim(0, 100)
                plt.legend()
                plt.grid(linestyle='-')
                init_reg *= 2
            plt.show()

        else:
            print("Argument was not valid")
    else:
        print("standard")

if __name__ == '__main__':
    main()