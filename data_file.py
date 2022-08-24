from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
all_data = load_iris()['data']
target = load_iris()['target']
two_dimensional_data = []
binary_target = []
for i in range(0,len(all_data)):
    current_mass = []
    if target[i] == 0 or target[i] == 1:
        binary_target.append(target[i])
        for j in range(0,2):
            current_mass.append(all_data[i][j])
        two_dimensional_data.append(current_mass)
#binary_target = np.where(np.array(binary_target) <= 0,-1,1)
X_train,X_test,y_train,y_test = train_test_split(two_dimensional_data,binary_target,train_size=0.8,random_state=1)
X_train = np.array(X_train).reshape(-1,len(X_test[0]))
X_test = np.array(X_test).reshape(-1,len(X_test[0]))
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)
