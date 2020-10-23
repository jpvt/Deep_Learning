import numpy as np

def f(x):
    return np.sin(x + np.sin(x)**2)

def generate_data(interval: tuple, k_steps: int, step_size: float):
    X = []
    y = []
    
    for n_step in range(interval[0], interval[1]):
        # Here we get the last K steps, making the interval [n_step - K, n_step)
        X.append(np.array(list(map(f, np.arange(n_step - k_steps, n_step, step_size)))))
        # And here, we get the following 3 steps [n_step, n_step+3), that represent the output of our network
        y.append(np.array(list(map(f, np.arange(n_step, n_step + 3, step_size)))))
    
    return X, y