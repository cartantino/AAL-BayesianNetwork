import matplotlib.pyplot as plt
import pandas as pd
import preprocessing
import seaborn as sns

#CORRELATION PLOT OF THE ORIGINAL DATASET
dataframe_or = preprocessing.load_dataset()

dataframe_or = dataframe_or.drop(['user','gender','age','classes'],axis=1)
ax_or = sns.heatmap(dataframe_or.corr(),annot = True, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200))

ax_or.set_xticklabels(
    ax_or.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')

plt.savefig('plot/corr_mat_original_dataset')

#CORRELATION MATRIX OF THE AFTER-MANIPULATED DATASET
dataframe_dis = preprocessing.load_data_discrete()

#REMOVING CLASSES ATTRIBUTE FROM THE DATAFRAME
dataframe_dis = dataframe_dis.drop(["sitting","sittingdown","standing","standingup","walking"], axis=1)
ax = sns.heatmap(dataframe_dis.corr(), annot = True, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200))

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')

plt.savefig('plot/corr_mat_discret_dataset')