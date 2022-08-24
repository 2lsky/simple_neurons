import numpy as np
import pandas as pd
from data_file import X_train
def Standardatization(X):
    new_X = np.ones((X.shape[0],X.shape[1]))
    for i in range(0,X.shape[1]):
        new_X[:,i] = (np.array(X[:,i])-np.array(X[:,i]).mean())/np.array(X[:,i]).std()
    return new_X
