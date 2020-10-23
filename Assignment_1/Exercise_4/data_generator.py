import numpy as np

def f(x):
    return np.sin(x + np.sin(x)**2)

def generate_data(interval: tuple, k_steps: int, step_size: float):
    X = []
    y = []
    
    for n_step in range(interval[0], interval[1]):
        # Aqui, pegamos os K passos anteriores
        X.append(np.array(list(map(f, np.arange(n_step - k_steps, n_step, step_size)))))
        # Já aqui, pegamos os 3 passos seguintes, que representariam a saída de nossa rede
        y.append(np.array(list(map(f, np.arange(n_step, n_step + 3, step_size)))))
    
    return X, y