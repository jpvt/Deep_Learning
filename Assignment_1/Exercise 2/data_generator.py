def generate_data(data_size, train_test_split=0.1):
    """
    Args : 
        data_size = number of instances to be produced
        train_val_split = percentage of data to be used on validation split
    Return:
        
    """
    X = np.zeros((data_size, 2))
    y = np.zeros((data_size, 1))

    for i in range(data_size):
        X[i] = np.random.randint(0,2,2)
        if X[i][0] != X[i][1]:
            y[i] = 1
        X[i] += np.random.uniform(-0.1,0.1, 2)
    
    test_split = round(data_size * (1 - train_test_split))

    X_train, y_train = X[:test_split], y[:test_split]
    X_test, y_test = X[test_split:], y[test_split:]

    return X_train, X_test, y_train, y_test