elif sys.argv[1] == "L1lambda":
    i = input("How many epochs? ")
    j = input("What learning rate? ")
    init_reg = 0.001
    for n in range(30):
        regularisation = 1, init_reg
        x, y, tx, ty = nn.neural_network(train_X, train_y, test_X, test_y, int(i), float(j), regularisation, cross_val)
        reg_l = "Reg Lambda=" + "{:.3f}".format(init_reg)
        plt.plot(x, y, label="train" + reg_l)
        plt.plot(tx, ty, label="test" + reg_l)
        plt.title("Basic Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j))
        plt.xlabel("Number of Epochs")
        plt.ylabel("Training Error")
        # plt.ylim(0, 100)
        plt.legend()
        plt.grid(linestyle='-')
        init_reg += 0.001
    plt.show()
elif sys.argv[1] == "L2lambda":
    i = input("How many epochs? ")
    j = input("What learning rate? ")
    init_reg = 0.001
    for i in range(7):
        regularisation = 2, init_reg
        x, y, tx, ty = nn.neural_network(train_X, train_y, test_X, test_y, int(i), float(j), regularisation, cross_val)
        reg_l = "Reg Lambda=" + "{:.2f}".format(init_reg)
        plt.plot(x, y, label="train" + reg_l)
        plt.plot(tx, ty, label="test" + reg_l)
        plt.title("Basic Linear Regression, Epochs=" + str(i) + ", Learning Rate=" + str(j))
        plt.xlabel("Number of Epochs")
        plt.ylabel("Training Error")
        # plt.ylim(0, 100)
        plt.legend()
        plt.grid(linestyle='-')
        init_reg *= 2
    plt.show()