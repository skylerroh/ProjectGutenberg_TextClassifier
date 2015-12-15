import pandas as pd
import numpy as np
import random
import time
from math import floor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
import path
from os.path import isfile, join
import pdb


OUTPUT_DIR = "/Users/Skyroh/Documents/Stat154/"


train_df = pd.read_csv('/Users/Skyroh/Documents/Stat154/Training_Output/doc_feature_top_word_frequencies_and_category_unique_words.csv', index_col = 0) #'/Users/fenglin/Desktop/stat154/Training_Output/doc_feature_matrix.csv')
#train_df.drop(train_df.columns[0], axis=1)
print(train_df.columns)


labels = train_df['tags'].values
print(labels)
features = np.nan_to_num(train_df.ix[:, train_df.columns != 'tags'].values)
print(features)

ntree_list = [1000]#[100, 250, 500, 1000]
depth = [None]
p = len(train_df.columns) - 1 #2000
n_features = [44]#[20, floor(p**(.5)), p//20, p//10, p//5, p//2]


test_df = pd.read_csv('/Users/Skyroh/Documents/Stat154/Training_Output/test_feature_top_word_frequencies_and_category_unique_words.csv', index_col = 0)
print(len(test_df.columns))
print(test_df.values)

truth = pd.read_csv('/Users/Skyroh/Github/Stat154FinalProject/Practice_label.csv', index_col = 0)
truth_index = truth.index
#truth = list(truth['category'])
#print(type(list(truth)))

np.random.RandomState()
#build rf with parameters from cv
rf = RandomForestClassifier(n_estimators = 500, max_features = 50, max_depth = None, class_weight = "auto", n_jobs = 3)
rf = rf.fit(features, labels)



from sklearn.externals import joblib
joblib.dump(rf, '/Users/Skyroh/Github/Stat154FinalProject/rf.pkl', compress = 3)
#rf = joblib.load('rf.pkl')
#features1 = test_df.ix[:, train_df.columns != 'tags'].values
prediction_array = pd.Series(rf.predict(np.nan_to_num(test_df.values)), index = test_df.index)
#prediction_array = prediction_array.ix[[str(i) for i in list(truth_index)]]
# print(type(prediction_array))
# print(prediction_array)
pred = list(prediction_array)
truth = list(truth['category'])
print(truth)
print(pred)
print(len(truth) == len(pred))
c0, c1, c2, c3 = 0, 0, 0, 0
t0, t1, t2, t3 = 0, 0, 0, 0
for i in range(len(test_df.index)):
	if truth[i] == 0:
		if truth[i] == pred[i]:
			c0 += 1
		t0 += 1
	elif truth[i] == 1:
		if truth[i] == pred[i]:
			c1 += 1
		t1 += 1
	elif truth[i] == 2:
		if truth[i] == pred[i]:
			c2 += 1
		t2 += 1
	elif truth[i] == 3:
		if truth[i] == pred[i]:
			c3 += 1
		t3 += 1
print([c0/t0, c1/t1, c2/t2, c3/t3])
print('overall:', (c0+c1+c2+c3)/(t0+t1+t2+t3))

form = {'category': list(prediction_array)}
final_output = pd.DataFrame(form)
final_output.index.name = 'id'
print(final_output)

final_output.to_csv(join(OUTPUT_DIR, "practice_predict.csv"))
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