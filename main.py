import matplotlib.pyplot as plt
from types_of_neurons import Perceptron,Adaline_SGD,Logistic_SGD
from data_file import X_train,X_test,y_train,y_test
from plot_module import plot_distribution
from transformation_of_data import Standardatization
import matplotlib.pyplot as plt
X_train = Standardatization(X_train)
X_test = Standardatization(X_test)
models = [Logistic_SGD(True,X_train,y_train,1000,50,0)]
for model in models:
    epoches,errors = model.fit()
    fig,ax = plt.subplots(ncols=1,nrows=1)
    ax.scatter(epoches,errors)
    plt.show()
    plot_distribution(X_train,y_train,model)
errors_1 = 0
for i in range(0,len(y_test)):
    if y_test[i] != models[0].predict(models[0].net_output(X_test[i])):
        errors_1 += 1
print(f'Logistic - {100 - errors_1*100/len(y_test)} %

