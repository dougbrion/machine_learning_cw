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
import run_linr as rl
import run_nn as nn

def main():

    seed = 13
    np.random.seed(seed)
    tf.set_random_seed(seed)

    learning_rate_list = [0.1, 0.001, 0.0001, 0.00001]
    epochs_list = [100, 500, 1000, 5000, 10000, 50000]
    cost_fn_list = [1, 2, 3, 4, 5]
    regularisation_list = [0, 1, 2]
    reg_param_list = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024]

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

        if sys.argv[1] == "linr":
            rl.run_linr(train_X, train_y, test_X, test_y, train_percent, cross_val)

        elif sys.argv[1] == "nn":
            nn.run_nn(train_X, train_y, test_X, test_y, train_percent, cross_val)

        elif sys.argv[1] == "linr_break":
            rl.linr_break(train_X, train_y, test_X, test_y, train_percent, cross_val)

        elif sys.argv[1] == "histogram":
            hp.histogram()

        elif sys.argv[1] == "plotter":
            print("Running Plotter")
            for i in epochs_list:
                for j in learning_rate_list:
                    for k in cost_fn_list:
                        for l in regularisation_list:
                            for m in reg_param_list:
                                
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