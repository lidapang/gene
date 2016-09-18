import numpy as np
import pandas as pd
from pprint import pprint
from prepare_data import PrepareData
from sklearn.ensemble import RandomForestClassifier


prepared_data = PrepareData()

labels = prepared_data.tag
raw_data = prepared_data.raw_data
training_data = raw_data[1:]
index_data = raw_data[0]
print(np.array(labels).shape)
print(np.array(training_data).shape)

# forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1, oob_score=True)
# forest.fit(training_data, labels)

# importances = forest.feature_importances_
# indices = np.argsort(importances)[::-1]

# print(forest.oob_score_)

# for f in range(len(index_data)):
#     print("%2d) %-*s %f" % (indices[f], 30, index_data[f], importances[indices[f]]))
import matplotlib.pyplot as plt

from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest, chi2

###############################################################################
# import some data to play with
# Add the noisy data to the informative features
X = np.array(training_data)
y = labels

###############################################################################

X_indices = np.arange(X.shape[-1])

###############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
selector = SelectKBest(chi2, k=100)
selector.fit(X, y)
scores = selector.scores_
scores /= scores.max()
# plt.bar(X_indices - .45, scores, width=.2,
#         label=r'Chi2 Score (scores)', color='g')
# indices = np.argsort(scores)[::-1]
# data2save = []
#
# for f in range(len(index_data)):
#     print("%2d) %-*s %f" % (indices[f], 30, index_data[indices[f]], scores[indices[f]]))
#     data2save.append([indices[f], index_data[indices[f]], scores[indices[f]]])
#
# result_df_2save = pd.DataFrame(data=data2save, columns=["index", "name", "importance"])
# result_df_2save.to_csv("./result/2_chi2.csv")

# Compare to the weights of an SVM
# clf = RandomForestClassifier(n_estimators=3000, n_jobs=-1)
# clf.fit(X, y)
#
# svm_weights = clf.feature_importances_ * 100
# svm_weights /= svm_weights.max()


clf_selected = RandomForestClassifier(n_estimators=3000, n_jobs=-1)
clf_selected.fit(selector.transform(X), y)

svm_weights_selected = clf_selected.feature_importances_ * 100
svm_weights_selected /= svm_weights_selected.max()

indices = np.argsort(svm_weights_selected)[::-1]

data2save = []

for f in range(0, 100):
    # print("%2d) %-*s %f" % (indices[f], 30, index_data[indices[f]], svm_weights_selected[indices[f]]))
    data2save.append([indices[f], index_data[indices[f]], svm_weights_selected[indices[f]]])

result_df_2save = pd.DataFrame(data=data2save, columns=["index", "name", "importance"])
result_df_2save.to_csv("./result/2_rf_selected.csv")