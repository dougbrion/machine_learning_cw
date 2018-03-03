import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sb

path = "../data/"
infile = "winequality-red.csv"

def dataset(_path, _infile):
    ds = pd.read_csv(_path + _infile, sep = ';')
    return ds

def tf_info():
    tf.reset_default_graph()
    # Check TF version
    print("Tensorflow Version: ", tf.__version__)

def calc_error(x, y):
    predictions = tf.add(b, tf.matmul(x, w))
    error = tf.reduce_mean(tf.square(y - predictions))
    return [predictions, error]

def data_visualise(_path, _infile):
    ds = pd.read_csv(_path + _infile, sep = ';')
    column_list = list(ds.columns)[0:-1]
    sb.pairplot(ds.loc[:,column_list], size=3)

def normalise(ds, cols):
    for col in cols:
        mean = ds.loc[:,col].mean()
        std_dev = np.std(ds.loc[:,col].values)
        ds.loc[:,col] = (ds.loc[:,col] - mean) / std_dev
    return ds

def rm_outliers(ds, thres, cols):
    for col in cols:
        mask = ds[col] > float(thres) * ds[col].std() + ds[col].mean()
        ds.loc[mask == True, col] = np.nan
        mean_prop = ds.loc[:,col].mean()
        ds.loc[mask == True, col] = mean_prop
    return ds

def data_info(_path, _infile):
    ds = pd.read_csv(_path + _infile, sep = ';')
    # print(ds.describe())
    # print(ds.corr())
    features_list = ds.columns.values[:-2]
    labels_column = ds.columns.values[-1]
    print("The features are: {}".format(features_list))
    print("The label column is: {}".format(labels_column))

# def data_convert(_path, _infile):
#     ds = read_csv(_path + _infile, sep = ';')
#     y = [0 if item == 'Good' else 1 for item in ds['category']]
#     print(y[0:5])
#     X = ds.drop(['quality', 'category'], axis=1).values
#     print(X)

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

# tf_info()
# data_info(path, infile)
# data_convert(path, infile)
# data_visualise(path, infile)

threshold = 5
ds = dataset(path, infile)
column_list = list(ds.columns)[:-1]
ds_norm = normalise(ds, column_list)
ds_no_outliers = rm_outliers(ds_norm, threshold, column_list[0:-1])
print("The range in wine quality is {0}".format(np.sort(ds_no_outliers['quality'].unique())))
ds_no_outliers.groupby(['quality']).count
ds_split = ds_no_outliers.copy()
bins = [3, 5, 8]
ds_split['category'] = pd.cut(ds_split.quality, bins, labels = ['bad', 'good'], include_lowest = True)
print("\nGood:\n", ds_split.loc[ds_split.loc[:,'category'] == 'good',['quality', 'category']].describe())
print("\nBad:\n", ds_split.loc[ds_split.loc[:,'category'] == 'bad',['quality', 'category']].describe())

ds_split.groupby(['category']).count()
ds_fixed = ds_split.copy()
ds_fixed.to_csv('../data/winequality-red-fixed.csv', index=False)

plot_data(path, infile)
