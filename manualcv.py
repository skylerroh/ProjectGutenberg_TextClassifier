import pandas as pd
import numpy as np
import time
from math import floor
from random import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
import pdb

def foo():
	train_df = pd.read_csv('/Users/Skyroh/Documents/Stat154/Training_Output/doc_feature_top_word_frequencies_100_to_2100.csv', index_col = 0)#doc_feature_top_wc_top_tfidf_var.csv', index_col = 0) 
	#train_df.drop(train_df.columns[0], axis=1)
	print(train_df.columns)

	labels = train_df['tags'].values
	print(labels)
	features = train_df.ix[:, train_df.columns != 'tags'].values
	print(features)

	#parameter list to test over
	ntree_list = [250]#[100, 250, 500, 1000]
	depth = [None]
	p = len(train_df.columns) - 1 #2000
	n_features = [44]#[20, floor(p**(.5)), p//20, p//10, p//5, p//2]


	accuracy = []
	feat_importances_total =np.array([0 for i in range(len(train_df.columns) - 1)])
	k = KFold(n=len(labels), n_folds=5, shuffle=True, random_state=None)
	for train_index, test_index in k:
		print("TRAIN:", len(train_index), "TEST:", len(test_index))
		X_train, X_test = features[train_index], features[test_index]
		y_train, y_test = labels[train_index], labels[test_index]
		rf = RandomForestClassifier(n_estimators = 250, max_features = 44, max_depth = None, class_weight = "auto", n_jobs = 3)
		rf = rf.fit(X_train, y_train)
		prediction_array = rf.predict(X_test)
		pred = list(prediction_array)
		y_test = list(y_test)
		#print(pred)
		#print(len(y_test) == len(pred))
		c0, c1, c2, c3 = 0, 0, 0, 0
		t0, t1, t2, t3 = 0, 0, 0, 0
		for i in range(len(X_test)):
			if y_test[i] == 0:
				if y_test[i] == pred[i]:
					c0 += 1
				t0 += 1
			elif y_test[i] == 1:
				if y_test[i] == pred[i]:
					c1 += 1
				t1 += 1
			elif y_test[i] == 2:
				if y_test[i] == pred[i]:
					c2 += 1
				t2 += 1
			elif y_test[i] == 3:
				if y_test[i] == pred[i]:
					c3 += 1
				t3 += 1
		print([c0/t0, c1/t1, c2/t2, c3/t3])
		overall = (c0+c1+c2+c3)/(t0+t1+t2+t3)
		print('overall:', overall)
		#score = rf.score(X_test, y_test)
		accuracy += [overall]
		#feat_importances_total = np.add(feat_importances_total, rf.feature_importances_)
		#plt.hist(rf.feature_importances_, bins = 100)
		#plt.show()
	print(np.mean(accuracy))
	#plt.hist(feat_importances_total, bins = 100)
	#plt.show()
#####################
#feature importance selection
	# features = train_df.ix[:, train_df.columns != 'tags']
	# features = train_df.ix[:, list((feat_importances_total > np.sort(feat_importances_total)[500]))]

	# features_update = features.values
	# print(features_update.shape)

	# accuracy = []
	# feat_importances_total =np.array([0 for i in range(len(train_df.columns) - 1)])
	# k = KFold(n=len(labels), n_folds =, shuffle=True, random_state=None)
	# for train_index, test_index in k:
	# 	print("TRAIN:", len(train_index), "TEST:", len(test_index))
	# 	X_train, X_test = features_update[train_index], features_update[test_index]
	# 	y_train, y_test = labels[train_index], labels[test_index]
	# 	rf = RandomForestClassifier(n_estimators = 250, max_features = 44, max_depth = None, class_weight = "auto", n_jobs = 3)
	# 	rf = rf.fit(X_train, y_train)
	# 	score = rf.score(X_test, y_test)
	# 	print(score)
	# 	accuracy += [score]
	# 	#plt.hist(rf.feature_importances_, bins = 100)
	# 	#plt.show()
	# print(np.mean(accuracy))
####################	



	# test_df = pd.read_csv('/Users/Skyroh/Documents/Stat154/Training_Output/test_feature_top_word_frequencies_100_to_2100.csv', index_col = 0)
	# print(len(test_df.columns))
	# print(test_df.values)

	# test_features = np.nan_to_num(test_df.ix[:, features.columns].values)
	# print(test_features.shape)
	# truth = pd.read_csv('/Users/Skyroh/Github/Stat154FinalProject/Practice_label.csv', index_col = 0)
	# truth_index = truth.index
	# #truth = list(truth['category'])
	# #print(type(list(truth)))

	# #build rf with parameters from cv
	# rf = RandomForestClassifier(n_estimators = 250, max_features = 44, max_depth = None, class_weight = "auto", n_jobs = 3)
	# rf = rf.fit(features_update, labels)

	# print(rf.predict(test_features))
	# #from sklearn.externals import joblib
	# #joblib.dump(rf, '/Users/Skyroh/Documents/Stat154/wc_and_tcidf_var_rf.pkl')

	# #features1 = test_df.ix[:, train_df.columns != 'tags'].values
	# prediction_array = pd.Series(rf.predict(test_features), index = test_df.index)
	# prediction_array = prediction_array.ix[[str(i) +'.txt' for i in list(truth_index)]]
	# pred = list(prediction_array)
	# truth = list(truth['category'])
	# print(pred)
	# print(len(truth) == len(pred))
	# c0, c1, c2, c3 = 0, 0, 0, 0
	# t0, t1, t2, t3 = 0, 0, 0, 0
	# for i in range(len(test_df.index)):
	# 	if truth[i] == 0:
	# 		if truth[i] == pred[i]:
	# 			c0 += 1
	# 		t0 += 1
	# 	elif truth[i] == 1:
	# 		if truth[i] == pred[i]:
	# 			c1 += 1
	# 		t1 += 1
	# 	elif truth[i] == 2:
	# 		if truth[i] == pred[i]:
	# 			c2 += 1
	# 		t2 += 1
	# 	elif truth[i] == 3:
	# 		if truth[i] == pred[i]:
	# 			c3 += 1
	# 		t3 += 1
	# print([c0/t0, c1/t1, c2/t2, c3/t3])
	# print('overall:', (c0+c1+c2+c3)/(t0+t1+t2+t3))
