import helpers as hp
import neural_network as nn
import sys
import numpy as np
import random
import tensorflow as tf

def run_nn(train_X, train_y, test_X, test_y, train_percent, cross_val):
    i = int(input("How many epochs? "))
    j = float(input("What learning rate? "))
    reg_level = int(input("What regularisation: 0 (None), 1 or 2? "))
    scale = 0
    reg_str = ""
    if reg_level != 0:
        scale = float(input("What scale for regularisation? "))
        reg_str = ", Regularisation=L" + str(reg_level) + ", Scale="  + str(scale)
    regularisation = reg_level, scale
    title = "Neural Network, Epochs=" + str(i) + ", Learning Rate=" + str(j) + "\nHidden ReLU layer, Output ReLU layer" + reg_str
    print("Running Neural Network")
    x, y, tx, ty = nn.neural_network(train_X, train_y, test_X, test_y, int(i), float(j), regularisation, cross_val)
    hp.plotter(title, x, y, tx, ty, train_percent)