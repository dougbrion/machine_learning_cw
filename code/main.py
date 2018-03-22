import helpers as hp
import neural_network as nn
import linear_regression as linr
import sys
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import run_linr as rl
import run_nn as nn
import plotters as plters

def seeder():
    seed = 13
    np.random.seed(seed)
    tf.set_random_seed(seed)

rate_list = [0.1, 0.01, 0.01, 0.1, 0.1]
reg_list = [[0,0],[0,0],[0,0],[0,0],[1,0.002]]
cost_list = [1,2,1,2,1]


def main():
    seeder()       
    if len(sys.argv) == 2:

        if sys.argv[1] == "linr":
            train_X, train_y, test_X, test_y, train_percent, cross_val = hp.intro()
            rl.run_linr(train_X, train_y, test_X, test_y, train_percent, cross_val)

        elif sys.argv[1] == "nn":
            train_X, train_y, test_X, test_y, train_percent, cross_val = hp.intro()
            nn.run_nn(train_X, train_y, test_X, test_y, train_percent, cross_val)

        elif sys.argv[1] == "linr_break":
            train_X, train_y, test_X, test_y, train_percent, cross_val = hp.intro()
            rl.linr_break(train_X, train_y, test_X, test_y, train_percent, cross_val)

        elif sys.argv[1] == "histogram":
            hp.histogram()

        elif sys.argv[1] == "linr_plotter":
            plters.linr_plotter()

        elif sys.argv[1] == "nn_plotter":
            plters.nn_plotter()

        elif sys.argv[1] == "linr_cv":
            train_X, train_y, test_X, test_y = hp.cv_intro()
            x, y, tx, ty = linr.linear_regression_params(train_X, train_y, test_X, test_y, 100, rate_list, cost_list, reg_list)
            hp.plotter("", x, y, tx, ty, 80)
            
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