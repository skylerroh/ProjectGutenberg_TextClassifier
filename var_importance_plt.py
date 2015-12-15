import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

train_df = pd.read_csv('/Users/Skyroh/Documents/Stat154/Training_Output/trainingset_feature_top_word_frequencies_and_category_unique_words.csv', index_col = 0)
featureImportance = rf.feature_importances_ / rf.feature_importances_.max()
sortedIdx = np.argsort(featureImportance)[2450:]
barPos = np.arange(sortedIdx.shape[0]) + .5

plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=8)

plt.barh(barPos, featureImportance[sortedIdx], align = 'center')
plt.yticks(barPos, train_df.ix[:, train_df.columns != 'tags'].columns[sortedIdx])
plt.xlabel('Variable Importance')
plt.subplots_adjust(left = .2, right = .9, top =.9, bottom = .1)
plt.show()