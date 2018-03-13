import helpers as hp
import linear_regression as linr
import logistic_regression as logr
import neural_network as nn
import support_vector_machine as svm
import sys

import matplotlib.pyplot as plt

learning_rate_list = [0.1, 0.001, 0.0001, 0.00001]
epochs_list = [100, 500, 1000, 5000, 10000, 50000]

def setup():
    i = input("How many epochs? ")
    j = input("What learning rate? ")
    title = "Basic Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j)
    return i, j, title

def plotter(x, y, filename, title, save):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Error")
    plt.grid(linestyle='-')
    if save == True:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def main():
    print("###############################")
    print("# Machine Learning Coursework #")
    print("######## Douglas Brion ########")
    print("###############################\n")

    if len(sys.argv) == 2:
        if sys.argv[1] == "testsuite":
            print("Running Test Suite")
        elif sys.argv[1] == "plotter":
            print("Running Plotter")
            for i in epochs_list:
                for j in learning_rate_list:
                    print("Starting Linear Regression " + str(i) + " " + str(j))
                    x, y = linr.run_linear_regression(i, j)
                    filename = "../figs/LINR_" + str(i) + "_" + str(j) + ".png"
                    title = "Basic Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j)
                    plotter(x, y, filename, title, True)
                    print("Finished Linear Regression " + str(i) + " " + str(j))

                    print("Starting Logistic Regression " + str(i) + " " + str(j))
                    x, y = logr.run_logistic_regression(i, j)
                    filename = "../figs/LOGR_" + str(i) + "_" + str(j) + ".png"
                    title = "Basic Logistic Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j)
                    plotter(x, y, filename, title, True)
                    print("Finished Logistic Regression " + str(i) + " " + str(j))

                    print("Starting Neural Network " + str(i) + " " + str(j))
                    x, y = nn.run_neural_network(i, j)
                    filename = "../figs/NN_" + str(i) + "_" + str(j) + ".png"
                    title = "Neural Network, Epochs=" + str(i) + ", Learning Rate=" + str(j) + "\nHidden ReLu layer, Output tanh layer"
                    plotter(x, y, filename, title, True)
                    print("Finished Neural Network " + str(i) + " " + str(j))

                    print("\nPlotted epochs " + str(i) + " and learning rate " + str(j) + "\n")

        elif sys.argv[1] == "linr":
            i = input("How many epochs? ")
            j = input("What learning rate? ")
            title = "Basic Linear Regression, Epochs=" + i + ", Learning Rate=" + j
            print("Running Linear Regression")
            x, y = linr.run_linear_regression(int(i), float(j))
            plotter(x, y, "", title, False)
        elif sys.argv[1] == "logr":
            i = input("How many epochs? ")
            j = input("What learning rate? ")
            title = "Basic Logistic Regression, Epochs=" + i + ", Learning Rate=" + j
            print("Running Logistic Regression")
            x, y = logr.run_logistic_regression(int(i), float(j))
            plotter(x, y, "", title, False)
        elif sys.argv[1] == "nn":
            i = input("How many epochs? ")
            j = input("What learning rate? ")
            title = "Neural Network, Epochs=" + i + ", Learning Rate=" + j + "\nHidden ReLu layer, Output tanh layer"
            print("Running Neural Network")
            x, y = nn.run_neural_network(int(i), float(j))
            plotter(x, y, "", title, False)
        elif sys.argv[1] == "svm":
            i = input("How many epochs? ")
            j = input("What learning rate? ")
            print("Running Support Vector Machine")
            x, y = svm.run_support_vector_machine(int(i), float(j))
        else:
            print("Argument was not valid")
    else:
        print("standard")

if __name__ == '__main__':
    main()