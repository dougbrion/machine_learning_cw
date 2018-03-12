import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
import helpers as hp

def dataset(_path, _infile):
    ds = pd.read_csv(_path + _infile, sep = ',')
    return ds

def data_info(ds):
    print(ds.describe())
    print(ds.corr())
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

def softmax_layer(X_tensor, num_units):
    num_inputs = X_tensor.get_shape()[1].value
    W = tf.Variable(tf.zeros([num_inputs, num_units]), name='W')
    b = tf.Variable(tf.zeros([num_units]), name='b')
    y = tf.nn.softmax(tf.matmul(X_tensor, W) + b)
    return y

def relu_layer(X_tensor, num_units):
    num_inputs = X_tensor.get_shape()[1].value
    W = tf.Variable(tf.random.uniform([num_features, num_units]), name='W')
    b = tf.Variable(tf.zeros([num_units]), name='b')
    y = tf.nn.relu(tf.matmul(X_tensor, W) + b, name='relu')
    return y

def define_cost_function(y, y_tensor, batch_size):
    cost = -tf.reduce_sum(y_tensor * tf.log(y), name='cross_entropy') / batch_size
    return cost

def train(cost, learning_rate):
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    return training_step

def compute_accuracy(y, y_tensor):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_tensor, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')
    return accuracy

def single_layer():
    # Create softmax layer
    with tf.name_scope("softmax") as scope:
        y_softmax = softmax_layer(X_placeholder, num_classes)

    # Define cost function
    with tf.name_scope("cost_function") as scope:
        global cost
        cost = define_cost_function(y_softmax, y_placeholder, batch_size)
        tf.summary.scalar("cost", cost)

    # Define training step
    with tf.name_scope("training") as scope:
        global training_step
        training_step = train(cost, learning_rate)

    # Calculate model accuracy
    with tf.name_scope("accuracy") as scope:
        global accuracy
        accuracy = compute_accuracy(y_softmax, y_placeholder)
        tf.summary.scalar("accuracy", accuracy)


ds = dataset(hp.PATH, hp.CATEGORIES)
data_info(ds)
y = [0 if item == 'good' else 1 for item in ds['category']]
print(y[0:5])
X = ds.drop(['quality', 'category'], axis=1).values
print(X)
y_one_hot = label_to_vec(y, num_classes=2)
print(y_one_hot[0:5])

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

learning_rate = 0.001
batch_size = X_train.shape[0] // 10
num_features = X_train.shape[1]
num_classes = 2
epochs = 1000
epochs_to_print = epochs // 10
hidden_layer_units = 20

X_placeholder = tf.placeholder(tf.float32, [None, num_features], name='X')
y_placeholder = tf.placeholder(tf.float32, [None, num_classes], name='y')

single_layer()

# Merge summaries for TensorBoard
merged_summaries = tf.summary.merge_all()

with tf.Session() as sess:

    log_directory = 'tmp/logs'
    summary_writer = tf.summary.FileWriter(log_directory, sess.graph)
    
    tf.global_variables_initializer().run()
    
    # average_cost = 0
    cost_sum = 0
    for i in range(epochs):
        
        X_batch, y_batch = random_sample(X_train, y_train, batch_size)
        feed_dict = {X_placeholder: X_batch, y_placeholder: y_batch}
        _, current_cost = sess.run([training_step, cost], feed_dict)
        cost_sum += current_cost
        
        # Print average cost periodically
        if i % epochs_to_print == 99:
            average_cost = cost_sum / epochs_to_print
            print("Epoch: {:4d}, average cost = {:0.3f}".format(i+1, average_cost))
            cost_sum = 0
    
    print('Finished model fitting.')
 
    # Calculate final accuracy
    X_batch, y_batch = random_sample(X_test, y_test, batch_size)
    feed_dict = {X_placeholder: X_test, y_placeholder: y_test}
    print("Final accuracy = {:0.3f}".format(sess.run(accuracy, feed_dict)))

print(num_features)