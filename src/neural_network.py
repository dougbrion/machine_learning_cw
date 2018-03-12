import tensorflow as tf
import helpers as hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.0001
epochs = 1000
batch_size = 100
step = 50

def load_ds(_path, _infile):
    ds = pd.read_csv(_path + _infile, sep = ',')
    return ds

def tanh_fn(_X, _inputs, _units):
    W = tf.Variable(tf.random_normal([_inputs, _units]), name='weight')
    b = tf.Variable(tf.random_normal([_units]), name='bias')
    y = tf.tanh(tf.matmul(_X, W) + b)
    return y, W

def neural_net(_X, _y, _sel):
    hidden_layer_nodes = 6
    inputs = hp.num_features(_X)

    hidden_layer, hidden_weight = tanh_fn(_X, inputs, hidden_layer_nodes)

    pred, weight = tanh_fn(hidden_layer, 1)