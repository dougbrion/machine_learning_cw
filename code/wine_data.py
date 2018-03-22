import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import helpers as hp

def load_ds(_path, _w, _r):
    w_ds = pd.read_csv(_path + _w, sep = ';')
    # r_ds = pd.read_csv(_path + _r, sep = ';')
    # ds = pd.concat([w_ds, r_ds])
    return w_ds

def normalise_ds(ds, cols):
    for col in cols:
        mean = ds.loc[:,col].mean()
        std_dev = np.std(ds.loc[:,col].values)
        ds.loc[:,col] = (ds.loc[:,col] - mean) / std_dev
    return ds

def rm_outliers_ds(ds, thres, cols):
    for col in cols:
        mask = ds[col] > float(thres) * ds[col].std() + ds[col].mean()
        ds.loc[mask == True, col] = np.nan
        mean_prop = ds.loc[:,col].mean()
        ds.loc[mask == True, col] = mean_prop
    return ds

def data_info(ds):
    features_list = ds.columns.values[:-1]
    labels_column = ds.columns.values[-1]
    print(ds.describe())
    print(ds.corr())
    print("The features are: {}".format(features_list))
    print("The label column is: {}".format(labels_column))

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

ds = load_ds(hp.PATH, hp.WHITE, hp.RED)
data_info(ds)
column_list = list(ds.columns)[:-1]
norm_ds = normalise_ds(ds, column_list)
no_outliers_ds = rm_outliers_ds(norm_ds, hp.THRESHOLD, column_list[0:-1])
print("The range in wine quality is {0}".format(np.sort(no_outliers_ds['quality'].unique())))
# no_outliers_ds.groupby(['quality']).count()
copy_ds = no_outliers_ds.copy()
# copy_ds.to_csv('../data/winequality-white-fixed.csv', index=False)
data_info(copy_ds)

# bins = [3, 5, 9]
# copy_ds['category'] = pd.cut(copy_ds.quality, bins, labels = ['bad', 'good'], include_lowest = True)
# print("\nGood:\n", copy_ds.loc[copy_ds.loc[:,'category'] == 'good',['quality', 'category']].describe())
# print("\nBad:\n", copy_ds.loc[copy_ds.loc[:,'category'] == 'bad',['quality', 'category']].describe())
# # copy_ds.groupby(['category']).count()
# fixed_ds = copy_ds.copy()
# fixed_ds.to_csv('../data/winequality-fixed-categories.csv', index=False)