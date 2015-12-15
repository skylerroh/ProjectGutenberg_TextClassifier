import pandas as pd
import numpy as np
import random
import time
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import pickle

train_df = pd.read_csv('/Users/Skyroh/Documents/Stat154/Training_Output/trainingset_feature_top_word_frequencies_and_category_unique_words.csv', index_col = 0)#doc_feature_top_wc_top_tfidf_var.csv', index_col = 0) 
#train_df.drop(train_df.columns[0], axis=1)
print(train_df.columns)

labels = train_df['tags'].values
print(labels)
features = np.nan_to_num(train_df.ix[:, train_df.columns != 'tags'].values)
print(features.shape)

np.random.seed(1234321)
rand = np.random.choice(len(features), size = len(features)//2, replace = False)
print(rand)
features_small = features[rand]
print(features_small.shape)
labels_small = labels[rand]


#parameter list to test over
krnl = ['rbf', 'sigmoid']
C_range = [5, 10, 50, 100, 200]# = 1e^-2, 1e^-1, 1, 10, 100 #[.1, 1, 10, 100, 1000]
gamma_range = [10, 50, 100, 200]# = 1e^-4, 1e^-2, 1, 100 #'auto' = 1/n features = 1/2500
param_grid = dict(gamma = gamma_range, C = C_range, kernel = krnl)


cv = StratifiedKFold(labels_small, n_folds = 5, random_state = 20151204)
grid = GridSearchCV(SVC(decision_function_shape = 'ovo'), param_grid=param_grid, cv=cv, verbose = 5, n_jobs = 3)
grid.fit(features_small, labels_small)


print(grid.grid_scores_)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

#write grid
with open('/Users/Skyroh/Github/Stat154FinalProject/svm_grid.pkl', "w") as fp:
    pickle.dump(grid, fp)

# Load model from file
with open('/Users/Skyroh/Github/Stat154FinalProject/svm_grid.pkl', "r") as fp:
    grid_load = pickle.load(fp)
