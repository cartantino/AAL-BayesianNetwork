import matplotlib.pyplot as plt
import pandas as pd
import preprocessing
import seaborn as sns
import numpy as np



#CORRELATION PLOT OF THE ORIGINAL DATASET
dataframe_or = preprocessing.load_dataset()
plt.close()
dataframe_or = dataframe_or.drop(['user','gender','age','classes'],axis=1)
ax_or = sns.heatmap(dataframe_or.corr(),linewidth = '0.1', cbar = True, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200))

ax_or.set_xticklabels(
    ax_or.get_xticklabels(),
    rotation=45,
    horizontalalignment='right') 

plt.savefig('plot/corr_mat_original_dataset')

#CORRELATION MATRIX OF THE DISCRETIZED DATASET
dataframe_dis = preprocessing.load_data_discrete()
plt.close()
#REMOVING CLASSES ATTRIBUTE FROM THE DATAFRAME
dataframe_dis = dataframe_dis.drop(["sitting","sittingdown","standing","standingup","walking"], axis=1)
ax = sns.heatmap(dataframe_dis.corr(),linewidth = '0.1',cbar = True, annot =True, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200))

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')

plt.savefig('plot/corr_mat_discret_dataset')




 #CORRELATION MATRIX OF THE MANIPULATED DATASET
 dataframe_man = preprocessing.load_data_sampled()
 plt.close()
 #REMOVING CLASSES ATTRIBUTE FROM THE DATAFRAME
 dataframe_man = dataframe_man.drop(["sitting","sittingdown","standing","standingup","walking"], axis=1)
 ax = sns.heatmap(dataframe_man.corr(),linewidth = '0.1',cbar = True, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200))

 ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')

plt.savefig('plot/corr_mat_sampled_dataset')
sns.set(style="ticks") 
#plot3 = sns.pairplot(dataframe_man, hue="classes")
#plt.savefig('plot/plot_fighi_e_dove_trovarli')
'''
# ALL THE POSSIBLE PLOTS
sns.pairplot(dataframe_dis)
plt.savefig('plot/mix_of_plot')

classes = ['sitting','sittingdown','standing','standingup','walking']
for clas in classes:
    data = dataframe_dis[dataframe_dis.clas == 1]
    sns.scatterplot(x=clas,  y='pitch1', data=data)
    plt.show()
'''


#MEAN ACCELERATION

# mean = np.mean(dataframe_man['total_accel_sensor_1'])
# plot = sns.lmplot(x="total_accel_sensor_1", y="total_accel_sensor_1", data=dataframe_man)
# plt.show()

