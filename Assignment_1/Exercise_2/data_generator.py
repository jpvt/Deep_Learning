import numpy as np
import math

def f(x):
    y = math.sin(math.pi * x) / (math.pi * x)
    return y

class DataGenerator():
    
    def generate_problem_A(self, opt = 'std'):
        """
        Function that generates the data for the XOR logic function:
        
           X | Y |
        A| 0 | 0 | 0
        B| 0 | 1 | 1
        C| 1 | 0 | 1
        D| 1 | 1 | 0
        
        
        """
        
        if opt == 'std':

            X = np.array([[0,0], [0,1], [1,0], [1,1]])
            y = np.array([[1,0],[0,1],[0,1],[1,0]])

            for i in [0,1,2,3]:
                print(f'{X[i]} => {y[i]}')

            X_train, X_test, y_train, y_test = X, X, y, y

            return X_train, X_test, y_train, y_test
        
        elif opt == 'noise':
            
            print('under construction')
            
        else:
            
            print('invalid option!')
             
        
        
        
        
        
    def generate_problem_B(self, n_samples = 1000, train_val_split = 0.1):
        
        X = np.random.uniform(0.01, 5, n_samples)
        y = np.array([f(Xi) for Xi in X])
        
        
        val_split = round(n_samples * (1 - train_val_split))

        X_train, y_train = X[:val_split].T, y[:val_split].T
        X_val, y_val = X[val_split:].T, y[val_split:].T
        return X_train, X_val, y_train, y_val