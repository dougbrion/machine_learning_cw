import helpers as hp
import linear_regression as linr
import neural_network as nn
import sys
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

def linr_plotter():
    learning_rate_list = [0.1, 0.001, 0.0001, 0.00001]
    epochs_list = [100, 500, 1000, 5000, 10000, 50000]
    cost_fn_list = [1, 2, 3, 4, 5]
    regularisation_list = [0, 1, 2]
    reg_param_list = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024]
    
    for train_percent in range(10, 100, 10):
        for i in epochs_list:
            for j in learning_rate_list:
                for k in cost_fn_list:
                    for l in regularisation_list:
                        for m in reg_param_list:

                            ds = hp.load_ds(hp.PATH, hp.FIXED)
                            X, y = hp.split(ds)

                            y_size = len(y)

                            training_size = int((train_percent / 100) * y_size)

                            train_X, train_y, test_X, test_y = hp.random_train_test(X, y, training_size)

                            loss_str = "L1"
                            if k == 1:
                                loss_str = "L1"
                            elif k == 2:
                                loss_str = "L2"
                            elif k == 3:
                                loss_str = "Elastic Net"
                            elif k == 4:
                                loss_str = "SVR"
                            elif k == 5:
                                loss_str = "Huber"
                            
                            reg_str = ""
                            if l != 0:
                                reg_str = ", Regularisation=L" + str(l) + ", Scale=" + str(m)
                            print("\nStarting Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j) + ", Loss Function=" + loss_str + reg_str + ", Training Percent=" + str(train_percent))
                            x, y, tx, ty = linr.linear_regression(train_X, train_y, test_X, test_y, i, j, k, [l, m], False, [1.,1.])
                            filename = "../figs/LINR_E" + str(i) + "_LR" + str(j) + "_LF" + loss_str + "_R" + reg_str + "_TP" + str(train_percent) + ".png"
                            title = "Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j) +"\n Loss Function=" + loss_str + reg_str
                            hp.plotter(title, x, y, tx, ty, train_percent, filename, True)
                            print("Finished Linear Regression.\n")

def nn_plotter():
    learning_rate_list = [0.1, 0.001, 0.0001, 0.00001]
    epochs_list = [100, 500, 1000, 5000, 10000, 50000]
    regularisation_list = [0, 1, 2]
    reg_param_list = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024]
    
    for train_percent in range(10, 100, 10):
        for i in epochs_list:
            for j in learning_rate_list:
                for l in regularisation_list:
                    for m in reg_param_list:
                        ds = hp.load_ds(hp.PATH, hp.FIXED)
                        X, y = hp.split(ds)

                        y_size = len(y)

                        training_size = int((train_percent / 100) * y_size)

                        train_X, train_y, test_X, test_y = hp.random_train_test(X, y, training_size)
                        
                        reg_str = ""
                        if l != 0:
                            reg_str = ", Regularisation=L" + str(l) + ", Scale=" + str(m)
                        print("\nStarting Neural Network, Epochs=" + str(i) + ", Learning Rate=" + str(j) + reg_str + ", Training Percent=" + str(train_percent))
                        x, y, tx, ty = nn.neural_network(train_X, train_y, test_X, test_y, i, j, [l, m], False, 0)
                        filename = "../figs/NN_E" + str(i) + "_LR" + str(j) + "_R" + reg_str + "_TP" + str(train_percent) + ".png"
                        title = "Neural Network, Epochs=" + str(i) + ", Learning Rate=" + str(j) +"\n" + reg_str
                        hp.plotter(title, x, y, tx, ty, train_percent, filename, True)
                        print("Finished Neural Network.\n")