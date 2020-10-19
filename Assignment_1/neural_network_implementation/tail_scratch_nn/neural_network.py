import numpy as np
import matplotlib.pyplot as plt
from .activation_functions import *

class DNN():
    def __init__(self):
        """
        Initializes your class:
            parameters : dictionary of parameters, which will store W and b through propagation.
            cache : dictionary of cache, which will be responsible for storing A and Z during the propagation.
            grads: dictionary of gradients, which will store all gradients computed during backprop.
            v : dictionary with momentum ewa estimates
            s : dictionary with RMSprop ewa estimates
        Args:
            No arguments taken.
        return:
            No return.
        """
        
        self.parameters = {}
        self.cache = {}
        self.grads = {}
        self.v = {}
        self.s = {}

    def fit(self, X_train, y_train, hidden=relu, output=softmax):
        """
        Args : 
            X_train = input data of shape (n_x, number_of_examples).
            y_train = label vector of shape (n_y, number_of_examples).
            hidden : passed as argument the function used on the hidden layers
            output : function used on output layer
        """
        self.X_train = X_train
        self.y_train = y_train
        self.m = X_train.shape[1]
        self.hidden = hidden # function passed as argument to be used on hidden layers
        self.output = output # function passed as argument to be used on output layers
        
        if self.output == sigmoid:
            self.output_derivative = sigmoid_derivative
        elif self.output == softmax:
            self.output_derivative = softmax_derivative
        else:
            print("output activation not recognized")
            return -1
        
        if self.hidden == relu:
            self.hidden_derivative = relu_derivative
        elif self.hidden == sigmoid:
            self.hidden_derivative = sigmoid_derivative
        else:
            print("hidden activation not recognized")
            return -1
    
    def initialize_parameters(self, dims, adam_optimizer=False):
        """
        Args:
            dims = dimensions of the network.
            
            Example:
                dims = [3,3,8]
                
                A network with input size = 3, hidden layer = 3 and output layer = 8.
                
                The first dimension on the list must always be the length of each example.
                The last dimension on the list must always be the length of each output example.
                
                In a case where X_train shape = (3, 4500) and y_train shape = (8, 4500), 4500 in
                each shape represents the number of examples.
                
                dims = [3, 8]
        Return:
            parameters : a dictionary containing all weights and biases intialized
                
        """
        self.L = len(dims)
        for l in range(1, self.L):
            self.parameters["W" + str(l)] = np.random.randn(dims[l], dims[l-1]) * 0.01
            self.parameters["b" + str(l)] = np.zeros((dims[l], 1))
            if adam_optimizer:
                self.v["VdW" + str(l)] = np.zeros((dims[l], dims[l-1]))
                self.v["Vdb" + str(l)] = np.zeros((dims[l], 1))
                self.s["SdW" + str(l)] = np.zeros((dims[l], dims[l-1]))
                self.s["Sdb" + str(l)] = np.zeros((dims[l], 1))
        return self.parameters
    
    def propagate(self, X):
        """
        Does the forward propagation of the network
        """
        A_prev = X
        self.cache[f"A{0}"] = A_prev
        for l in range(1, self.L):
            
            Z = np.dot(self.parameters[f"W{l}"], A_prev) + self.parameters[f"b{l}"]

            if l == self.L - 1:
                A = self.output(Z)
            else:
                A = self.hidden(Z)

            self.cache[f"Z{l}"] = Z
            self.cache[f"A{l}"] = A
            
            A_prev = A
        
        self.y_hat = A

    def predict(self, X):
        """
        Predicts the value using the propagate function
        
        Args:
            X : data to be used on prediction
        Return:
            y_hat : data predicted
        """
        self.propagate(X)
        return self.y_hat
    
    def compute_cost(self):
        pred = self.y_hat.T
        real = self.y_train.T
        n_samples = real.shape[0]
        logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
        cost = np.sum(logp)/(n_samples)
        return cost

    def loss(self):
        res = self.y_hat - self.y_train
        return res

    def backprop(self):
        dA = self.loss()
        
        dZ = dA * self.output_derivative(self.cache[f"Z{self.L - 1}"])
        
        self.grads[f"dW{self.L - 1}"] = 1/self.m * (np.dot(dZ, self.cache[f"A{self.L - 2}"].T))
        self.grads[f"db{self.L - 1}"] = 1/self.m * np.sum(dZ, axis=1, keepdims=True)
        
        
        for l in reversed(range(1, self.L - 1)):
            self.grads[f"dA_prev{l}"] = np.dot(self.parameters[f"W{l + 1}"].T,dZ)
            dZ = self.grads[f"dA_prev{l}"] * self.hidden_derivative(self.cache[f"Z{l}"])
            self.grads[f"dW{l}"] = 1/self.m * (np.dot(dZ, self.cache[f"A{l - 1}"].T))
            self.grads[f"db{l}"] = 1/self.m * np.sum(dZ, axis=1, keepdims=True)
    
    def update_grads_adam(self, t, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        """
        ADAM -> Adaptive Moment estimation
        Args:
            t : epoch number
            learning_rate : learning rate chosed to upgrade weights
            beta1 : exponentially weighted average used on v (momentum), beta1 = 0.9 (recommended on paper) is approx 10 days ewa
            beta1 : exponentially weighted average used on s (RMSprop), beta2 = 0.999 (recommended on paper)
            epsilon : term to prevent division by zero
        """
        
        v_biasCorrected = {}
        s_biasCorrected = {}
        self.learning_rate = learning_rate
        
        for l in reversed(range(1, self.L)):
            # moving average of the gradients
            self.v[f"VdW{l}"] = beta1 * self.v[f"VdW{l}"] + (1 - beta1)* self.grads[f"dW{l}"]
            self.v[f"Vdb{l}"] = beta1 * self.v[f"Vdb{l}"] + (1 - beta1)* self.grads[f"db{l}"]

            v_biasCorrected[f"VdW{l}"] = self.v[f"VdW{l}"]/(1 - beta1 ** t) # bias correction to the first updates
            v_biasCorrected[f"Vdb{l}"] = self.v[f"Vdb{l}"]/(1 - beta1 ** t) # bias correction

            self.s[f"SdW{l}"] = beta2 * self.s[f"SdW{l}"] + (1 - beta2) * np.square(self.grads[f"dW{l}"])
            self.s[f"Sdb{l}"] = beta2 * self.s[f"Sdb{l}"] + (1 - beta2) * np.square(self.grads[f"db{l}"])
                                                                                             
            s_biasCorrected[f"SdW{l}"] = self.s[f"SdW{l}"]/(1 - beta2 ** t) # bias correction to the first updates
            s_biasCorrected[f"Sdb{l}"] = self.s[f"Sdb{l}"]/(1 - beta2 ** t) # bias correction
            
            self.parameters[f"W{l}"] -= self.learning_rate * (v_biasCorrected[f"VdW{l}"])/(np.sqrt(s_biasCorrected[f"SdW{l}"]) + epsilon)
            self.parameters[f"b{l}"] -= self.learning_rate * (v_biasCorrected[f"Vdb{l}"])/(np.sqrt(s_biasCorrected[f"Sdb{l}"]) + epsilon)
                                                                                               
    def update_grads_gd(self, learning_rate = 0.01):
        """
        Args:
            learning_rate : learning rate chosed to upgrade weights
        """
        self.learning_rate = learning_rate
        for l in reversed(range(1, self.L)):
            self.parameters[f"W{l}"] -= self.learning_rate * (self.grads[f"dW{l}"])
            self.parameters[f"b{l}"] -= self.learning_rate * (self.grads[f"db{l}"])

    def train(self, dims, learning_rate = 0.01, iterations = 1000, adam_optimizer=False):
        if iterations > 100:
            printing_interval = round(iterations * 0.01)
        else:
            printing_interval = 1
        self.initialize_parameters(dims, adam_optimizer=adam_optimizer)
        costs = []
        for i in range(iterations):
            self.propagate(self.X_train)
            cost = self.compute_cost()
            if i % printing_interval == 0:
                print(f"epoch {i} : {cost}")
            costs.append(cost)
            self.backprop()
            if adam_optimizer:
                self.update_grads_adam(t=i+1, learning_rate=learning_rate)
            else:
                self.update_grads_gd(learning_rate = learning_rate)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()