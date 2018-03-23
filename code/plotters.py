import helpers as hp
import linear_regression as linr
import neural_network as nn
import sys
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

def linr_plotter():
    learning_rate_list = [0.05]
    epochs_list = [500]
    cost_fn_list = [1, 2, 4, 5]
    # cost_fn_list = [3]

    regularisation_list = [3]
    # reg_param_list = [0.0001, 0.001, 0.01, 0.1]
    lamb = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0]
    alpha = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0]

    for i in epochs_list:
        for j in learning_rate_list:
            for k in cost_fn_list:
                for l in lamb:
                    for m in alpha:

                        ds = hp.load_ds(hp.PATH, hp.WHITEFIXED)
                        X, y = hp.split(ds)

                        y_size = len(y)

                        training_size = int((100 / 100) * y_size)

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
                        
                        reg_str = " Elastic Net: " + "α=" + str(m) + " λ=" + str(l)
                        # if l != 0:
                            # reg_str = ", Regularisation=L" + str(l) + ", Scale=" + str(m)
                        el = [l, m]
                        reg = [3,0.0]
                        print("\nStarting Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j) + ", Loss Function=" + loss_str + reg_str + "\n10-fold Cross Validation")
                        x, y, tx, ty = linr.linear_regression(train_X, train_y, test_X, test_y, i, j, k, reg, True, el)
                        filename = "../figs/WHITE_ELASTIC_LINR_E" + str(i) + "_LR" + str(j) + "_LF" + str(k) + "_R" + str(l) + "_S" + str(m) + ".png"
                        title = "White Wine\nLinear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j) +"\n Loss Function=" + loss_str + reg_str
                        hp.plotter(title, x, y, tx, ty, 90, filename, True)
                        print("Finished Linear Regression.\n")
    
    # for i in epochs_list:
    #     for j in learning_rate_list:
    #         for k in cost_fn_list:
    #             for l in regularisation_list:
    #                 for m in reg_param_list:

    #                     ds = hp.load_ds(hp.PATH, hp.FIXED)
    #                     X, y = hp.split(ds)

    #                     y_size = len(y)

    #                     training_size = int((100 / 100) * y_size)

    #                     train_X, train_y, test_X, test_y = hp.random_train_test(X, y, training_size)

    #                     loss_str = "L1"
    #                     if k == 1:
    #                         loss_str = "L1"
    #                     elif k == 2:
    #                         loss_str = "L2"
    #                     elif k == 3:
    #                         loss_str = "Elastic Net"
    #                     elif k == 4:
    #                         loss_str = "SVR"
    #                     elif k == 5:
    #                         loss_str = "Huber"
                        
    #                     reg_str = ""
    #                     if l != 0:
    #                         reg_str = ", Regularisation=L" + str(l) + ", Scale=" + str(m)
    #                     reg = [l,m]
    #                     print("\nStarting Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j) + ", Loss Function=" + loss_str + reg_str + "\n10-fold Cross Validation")
    #                     x, y, tx, ty = linr.linear_regression(train_X, train_y, test_X, test_y, i, j, k, reg, True)
    #                     filename = "../figs/RED_LINR_E" + str(i) + "_LR" + str(j) + "_LF" + str(k) + "_R" + str(l) + "_S" + str(m) + ".png"
    #                     title = "Red Wine\nLinear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j) +"\n Loss Function=" + loss_str + reg_str
    #                     hp.plotter(title, x, y, tx, ty, 90, filename, True)
    #                     print("Finished Linear Regression.\n")

def nn_plotter():
    learning_rate_list = [0.5, 0.1, 0.01, 0.001]
    epochs_list = [500]
    regularisation_list = [0, 1, 2]
    reg_param_list = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0]
    
    for i in epochs_list:
        for j in learning_rate_list:
            for l in regularisation_list:
                for m in reg_param_list:
                    ds = hp.load_ds(hp.PATH, hp.FIXED)
                    X, y = hp.split(ds)

                    y_size = len(y)

                    training_size = int((100 / 100) * y_size)

                    train_X, train_y, test_X, test_y = hp.random_train_test(X, y, training_size)
                    
                    reg_str = ""
                    if l != 0:
                        reg_str = "Regularisation=L" + str(l) + ", Scale=" + str(m)
                    print("\nStarting Neural Network, Epochs=" + str(i) + ", Learning Rate=" + str(j) + reg_str + "\n10-fold Cross Validation")
                    x, y, tx, ty = nn.neural_network(train_X, train_y, test_X, test_y, i, j, [l, m], True, 0)
                    filename = "../figs/RED_NN_E" + str(i) + "_LR" + str(j) + "_R" + str(l) + "_S" + str(m) + ".png"
                    title = "Red Wine\nNeural Network 10-fold Cross Validation, Epochs=" + str(i) + ", Learning Rate=" + str(j) +"\n Hidden ReLU, Output ReLU\n" + reg_str
                    hp.plotter(title, x, y, tx, ty, 90, filename, True)
                    print("Finished Neural Network.\n")