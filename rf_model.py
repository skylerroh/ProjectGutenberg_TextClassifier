import pandas as pd
import numpy as np
import random
import time
from math import floor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold

train_df = pd.read_csv('/Users/Skyroh/Documents/Stat154/Training_Output/doc_feature_top_wc_top_tfidf_var.csv', index_col = 0) #'/Users/fenglin/Desktop/stat154/Training_Output/doc_feature_matrix.csv')
#train_df.drop(train_df.columns[0], axis=1)
print(train_df.columns)

labels = train_df['tags'].values
print(labels)
features = train_df.ix[:, train_df.columns != 'tags'].values
print(features)

ntree_list = [250]#[100, 250, 500, 1000]
depth = [None]
p = len(train_df.columns) - 1 #2000
n_features = [44]#[20, floor(p**(.5)), p//20, p//10, p//5, p//2]

#oob_score = [[[0 for k in range(len(n_features))] for j in range(len(depth))] for i in range(len(ntree_list))]

# for i in range(len(ntree_list)):
#     for j in range(len(depth)):
#         for k in range(len(n_features)):
#             ntree = ntree_list[i]
#             deep = depth[j]
#             m = n_features[k]
#             print(ntree, deep, m)

#             random.seed(20151129)
#             rf = RandomForestClassifier(n_estimators = ntree, max_features = m, max_depth = deep, min_samples_split = 5, class_weight = "auto", n_jobs = 3, oob_score = True)
#             rf = rf.fit(features, labels)
#             oob_score[i][j][k] = rf.oob_score_
#             print(rf.oob_score_)
#             #print(rf.feature_importances_) #25 most important features

# print(oob_score)
random.seed()
#cv 5-fold, running oob first to shrink number of parameters to cv over
rf_score = [[[0 for k in range(len(n_features))] for j in range(len(depth))] for i in range(len(ntree_list))]
for i in range(len(ntree_list)):
    for j in range(len(depth)):
	    for k in range(len(n_features)):
	        ntree = ntree_list[i]
	        deep = depth[j]
	        m = n_features[k]
	        print('ntree:', ntree, '  depth:', deep, '  m_feat:', m)
			#print(ntree, deep, m)
	        start_time = time.time()
	        #random.seed(r)#20151129)
	        rf = RandomForestClassifier(n_estimators = ntree, max_features = m, max_depth = deep, class_weight = "auto", n_jobs = -1, oob_score = False)
	        #random.seed(r + 124564)
	        rf_score[i][j][k] = cross_val_score(rf, features, labels, scoring = 'accuracy', cv = 5, n_jobs = -1).mean()
	        print(rf_score[i][j][k])
	        elapsed_time = time.time() - start_time
	        print('elapsed_time(min):', elapsed_time/60)
	        print()
	        print()


print(rf_score)
