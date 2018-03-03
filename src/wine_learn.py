import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sb

path = "../data/"
infile = "winequality-red-fixed.csv"

learning_rate = 0.001
batch_size = X_train.shape[0] // 10
num_features = X_train.shape[1]
num_classes = 2
epochs = 1000
epochs_to_print = epochs // 10
hidden_layer_units = 20

def dataset(_path, _infile):
    ds = pd.read_csv(_path + _infile, sep = ',')
    return ds

def tf_info():
    tf.reset_default_graph()
    # Check TF version
    print("Tensorflow Version: ", tf.__version__)

def calc_error(x, y):
    predictions = tf.add(b, tf.matmul(x, w))
    error = tf.reduce_mean(tf.square(y - predictions))
    return [predictions, error]

def data_info(ds):
    # print(ds.describe())
    # print(ds.corr())
    features_list = ds.columns.values[:-2]
    labels_column = ds.columns.values[-1]
    print("The features are: {}".format(features_list))
    print("The label column is: {}".format(labels_column))

def label_to_vec(labels, num_classes=2):
    labels_one_hot = []
    for label in labels:
        indices = [1] * num_classes
        indices[label] = 0
        labels_one_hot.append(indices)
    return labels_one_hot

def plot_data(_path, _infile):
    ds = pd.read_csv(_path + _infile, sep = ';')
    values = ds.values
    groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    curr_plot = 1
    plt.figure()
    for group in groups:
        plt.subplot(len(groups), 1, curr_plot)
        plt.plot(values[:,group])
        plt.title(ds.columns[group], y = 0.5, loc="right")
        curr_plot += 1
    plt.show()

def random_sample(X, y, sample_size):  
    y_size = len(y)
    index_sample = np.random.choice(y_size, sample_size, replace=False)
    y_array = np.array(y)
    X_batch = X[index_sample]
    y_batch = y_array[index_sample] 
    return X_batch, y_batch

ds = dataset(path, infile)
data_info(ds)
y = [0 if item == 'good' else 1 for item in ds['category']]
print(y[0:5])
X = ds.drop(['quality', 'category'], axis=1).values
print(X)
y_one_hot = label_to_vec(y, num_classes=2)
print(y_one_hot[0:5])

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

X_placeholder = tf.placeholder(tf.float32, [None, num_features], name='X')
y_placeholder = tf.placeholder(tf.float32, [None, num_classes], name='y')