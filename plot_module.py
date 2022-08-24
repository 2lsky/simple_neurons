import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def plot_distribution(data,target,model):
    fig,ax = plt.subplots(nrows=1,ncols=1)
    dataframe = pd.DataFrame(data=data,columns = [f'feature_{i}' for i in range(1,data.shape[1]+1)])
    dataframe['target'] = target
    step_for_x = (dataframe['feature_1'].max() - dataframe['feature_1'].min())/100
    step_for_y = (dataframe['feature_2'].max() - dataframe['feature_2'].min())/100
    x_points_for_separation_areas = [dataframe['feature_1'].min() + step_for_x * i for i in range(0,101)]
    y_points_for_separation_areas = [dataframe['feature_2'].min() + step_for_y * i for i in range(0, 101)]
    predicts = np.ones((len(x_points_for_separation_areas),len(y_points_for_separation_areas)))
    x,y = np.meshgrid(x_points_for_separation_areas,y_points_for_separation_areas)
    for i in range(0,len(x_points_for_separation_areas)):
        for j in range(0,len(y_points_for_separation_areas)):
            predicts[i][j] = model.predict(model.net_output(np.array([x_points_for_separation_areas[i],y_points_for_separation_areas[j]])))
    cp = ax.contourf(x, y, predicts.T)
    fig.colorbar(cp)
    sns.scatterplot(data = dataframe,x='feature_1',y='feature_2',hue='target',ax=ax)
    plt.show()
