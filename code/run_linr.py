import helpers as hp
import linear_regression as linr
import sys
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

def run_linr(train_X, train_y, test_X, test_y, train_percent, cross_val):
    i = int(input("How many epochs? "))
    j = float(input("What learning rate? "))
    cost_fn = int(input("Would you like a 1 (L1), 2 (L2), 3 (Elastic Net), 4 (SVM), 5 (Huber Loss) cost function? "))
    scale = 0
    reg_str = ""
    params = [1.,1.]
    reg_level = 0
    cross = ""
    if cost_fn != 3:
        reg_level = int(input("What regularisation: 0 (None), 1 or 2? "))
        if reg_level != 0:
            scale = float(input("What scale for regularisation? "))
            reg_str = ", Regularisation=L" + str(reg_level) + ", Scale="  + str(scale)
    regularisation = reg_level, scale
    loss_str = "L1"
    if cost_fn == 1:
        loss_str = "L1"
    elif cost_fn == 2:
        loss_str = "L2"
    elif cost_fn == 3:
        loss_str = "Elastic Net"
        param1 = float(input("What Elastic Net Parameter 1, (Default 1.) "))
        param2 = float(input("What Elastic Net Parameter 2, (Default 1.) "))
        params = [param1, param2]
    elif cost_fn == 4:
        loss_str = "SVR"
    elif cost_fn == 5:
        loss_str = "Huber"
    if cross_val == True:
        cross = "\n10-fold Cross Validation"
        train_percent = 90
    title = "Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j) +"\n Loss Function=" + loss_str + reg_str + cross
    print("Running Linear Regression")
    x, y, tx, ty = linr.linear_regression(train_X, train_y, test_X, test_y, i, j, cost_fn, regularisation, cross_val, params)
    hp.plotter(title, x, y, tx, ty, train_percent)

def linr_break(train_X, train_y, test_X, test_y, train_percent, cross_val):
    i = 100
    j = 0.32
    incr = 0.01
    regularisation = [0, 0]
    params = [1.,1.]
    colors = ['grey', 'black']
    for n in range(0,2):
        x, y, tx, ty = linr.linear_regression(train_X, train_y, test_X, test_y, i, j, 2, regularisation, cross_val, params)
        learning_rate = "Learning Rate=" + "{:.2f}".format(j)
        plt.plot(x, y, label=learning_rate, color=colors[n])
        plt.title("Basic Linear Regression, Epochs=100, Learning Rate=variable\n L2 loss function")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Training Error (L2 loss)")
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(linestyle='-')
        j += incr
        print(n)
    plt.show()