from data_file import binary_target,two_dimensional_data
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
### only for 1 and -1 classification
class Perceptron():
    def __init__(self,lambda_parametr,X,git y,n_iter,random_state):
        self.lambda_parametr = lambda_parametr
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.random_state = random_state
        self.rgen = np.random.RandomState(self.random_state)
        self.weights = self.rgen.normal(0, 0.01, X.shape[1] + 1).reshape(-1, 1)
    def net_output(self,x_for_predict):
        return np.dot(x_for_predict,self.weights[1:]) + self.weights[0]
    def predict(self,value):
        return 1 if value > 0 else -1
    def fit(self):
        errors = []
        for i in range(0,self.n_iter):
            num_of_errors = 0
            for j in range(0,len(self.y)):
                delta_weights = np.array(self.lambda_parametr * (self.y[j] - self.predict(self.net_output(self.X[j]))) * self.X[j]).reshape(-1,1)
                delta_bias = self.lambda_parametr * (self.y[j] - self.predict(self.net_output(self.X[j])))
                if delta_bias != [0]:
                    num_of_errors += 1
                self.weights[1:] = self.weights[1:] + delta_weights
                self.weights[0] = self.weights[0] + delta_bias
            errors.append(num_of_errors)
        return range(1,self.n_iter+1),errors

### only for 1 and -1 classification
class Adaline_SGD():
    def __init__(self,index_of_lambda_parametr,X,y,n_iter,batch_size,random_state):
        n = 1
        self.index_of_lambda_parametr = index_of_lambda_parametr
        if self.index_of_lambda_parametr:
            self.lambda_parametr = float(input())
        else:
            self.c_1 = float(input())
            self.c_2 = float(input())
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.random_state = random_state
        self.rgen = np.random.RandomState(self.random_state)
        self.weights = self.rgen.normal(0,0.01,X.shape[1] + 1).reshape(-1,1)
        self.batch_size = batch_size
    def linear_activation(self,value):
        return value
    def net_output(self,x):
        return self.linear_activation((np.dot(x, self.weights[1:]) + self.weights[0]))
    def predict(self,output_value):
        return 1 if output_value > 0 else -1
    def shuffle(self):
        r = self.rgen.permutation(len(self.y))
        return self.X[r],self.y[r]
    def fit(self):
        avg_mse_for_batch = []
        for i in range(0,self.n_iter):
            X,y = self.shuffle()
            if self.index_of_lambda_parametr:
                lambda_parametr = self.lambda_parametr
            else:
                lambda_parametr = self.c_1/(self.c_2 + i)
            print(f'Epoch {i+1}')
            mse_for_batch = []
            for j in [0 + (self.batch_size) * (k) for k in range(0,len(y) // self.batch_size + 1)]:
                if (self.batch_size-1)+j < len(y):
                    end = (self.batch_size-1)+j
                else:
                    end = len(y) - 1
                #print((j,(self.batch_size-1)+j))
                errors = y[j:end] - self.net_output(X[j:end])
                delta_weights = (lambda_parametr * np.sum(errors * X[j:end],axis=0)).reshape(-1,1)
                delta_bias = lambda_parametr * np.sum(errors,axis=0)
                self.weights[1:] = self.weights[1:] + delta_weights
                self.weights[0] = self.weights[0] + delta_bias
                mse_for_batch.append(0.5 * np.square(errors).sum())
            avg_mse_for_batch.append(np.array(mse_for_batch).mean())
        return range(1,self.n_iter+1),avg_mse_for_batch

### only for 1 and 0 classification
class Logistic_SGD():
    def __init__(self,index_of_lambda_parametr,X,y,n_iter,batch_size,random_state):
        n = 1
        self.index_of_lambda_parametr = index_of_lambda_parametr
        if self.index_of_lambda_parametr:
            self.lambda_parametr = float(input())
        else:
            self.c_1 = float(input())
            self.c_2 = float(input())
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.random_state = random_state
        self.rgen = np.random.RandomState(self.random_state)
        self.weights = self.rgen.normal(0,0.01,X.shape[1] + 1).reshape(-1,1)
        self.batch_size = batch_size
    def sigmoid_activation(self,value):
        return 1/(1+np.exp(-value))
    def net_output(self,x):
        return self.sigmoid_activation(np.dot(x, self.weights[1:]) + self.weights[0])
    def predict(self,output_value):
        return 1 if output_value >= 0.5 else 0
    def shuffle(self):
        r = self.rgen.permutation(len(self.y))
        return self.X[r],self.y[r]
    def fit(self):
        avg_log_errors_for_batch = []
        for i in range(0,self.n_iter):
            X,y = self.shuffle()
            if self.index_of_lambda_parametr:
                lambda_parametr = self.lambda_parametr
            else:
                lambda_parametr = self.c_1/(self.c_2 + i)
            print(f'Epoch {i+1}')
            log_err_for_batch = []
            for j in [0 + (self.batch_size) * (k) for k in range(0,len(y) // self.batch_size + 1)]:
                if (self.batch_size-1)+j < len(y):
                    end = (self.batch_size-1)+j
                else:
                    end = len(y) - 1
                log_err = (-y[j:end]*np.log(self.net_output(X[j:end]))-(1-y[j:end])*(np.log(1-self.net_output(X[j:end]))))
                errors = y[j:end] - self.net_output(X[j:end])
                delta_weights = (lambda_parametr * np.sum(errors * X[j:end],axis=0)).reshape(-1,1)
                delta_bias = lambda_parametr * np.sum(errors,axis=0)
                self.weights[1:] = self.weights[1:] + delta_weights
                self.weights[0] = self.weights[0] + delta_bias
                log_err_for_batch.append(np.array(log_err).mean())
            avg_log_errors_for_batch.append(np.array(log_err_for_batch).mean())
        return range(1,self.n_iter+1),avg_log_errors_for_batch


