import pandas as pd
import numpy as np
import random
import time
from math import floor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold


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


test_df = pd.read_csv('/Users/Skyroh/Documents/Stat154/Training_Output/test_feature_matrix_top_wc_top_tfidf_var.csv', index_col = 0)
print(len(test_df.columns))
print(test_df.values)

truth = pd.read_csv('/Users/Skyroh/Github/Stat154FinalProject/example_label_kaggle.csv', index_col = 0)
truth = list(truth['category'])
#print(type(list(truth)))
print(list(truth))


#build rf with parameters from cv
rf = RandomForestClassifier(n_estimators = 250, max_features = 44, max_depth = None, min_samples_split = 5, class_weight = "auto", n_jobs = 3)
rf = rf.fit(features, labels)



#from sklearn.externals import joblib
#joblib.dump(rf, '/Users/Skyroh/Documents/Stat154/wc_and_tcidf_var_rf.pkl')

#features1 = test_df.ix[:, train_df.columns != 'tags'].values
pred = list(rf.predict(test_df.values))
print(pred)
print(len(truth) == len(pred))
correct = 0
for i in range(len(truth)):
	if truth[i] == pred[i]:
		correct += 1
print(correct/len(truth))

# rf_score = [[[0 for k in range(len(n_features))] for j in range(len(depth))] for i in range(len(ntree_list))]
# for i in range(len(ntree_list)):
#     for j in range(len(depth)):
# 	    for k in range(len(n_features)):
# 	        ntree = ntree_list[i]
# 	        deep = depth[j]
# 	        m = n_features[k]
# 	        print('ntree:', ntree, '  depth:', deep, '  m_feat:', m)
# 			#print(ntree, deep, m)
# 	        start_time = time.time()