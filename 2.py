import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pprint import pprint
from prepare_data import PrepareData
from sklearn import datasets, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest, chi2, RFECV
from sklearn_patch import RandomForestClassifierWithCoef

if __name__ == "__main__":
    prepared_data = PrepareData()

    labels = prepared_data.tag
    raw_data = prepared_data.raw_data
    X = np.array(raw_data[1:])
    y = labels
    index_data = raw_data[0]

    X_indices = np.arange(X.shape[-1])

    ## Chi2 test
    selector = SelectKBest(chi2, k=20)
    selector.fit(X, y)
    scores = selector.scores_
    pvalues = -np.log10(selector.pvalues_)
    scores /= scores.max()
    pvalues /= pvalues.max()

    ## Plot chi2 scores
    plt.figure()
    plt.bar(X_indices - .45, scores, width=.2,
            label=r'Chi2 Score (scores)', color='g')
    plt.savefig('./result/2_chi2_scores.png')
    plt.close()
    ## Plot chi2 pvalues
    plt.figure()
    plt.bar(X_indices - .45, pvalues, width=.2,
            label=r'Chi2 Score ($-Log(p_{value})$)', color='g')
    plt.savefig('./result/2_chi2_pvalues.png')
    plt.close()

    ## Save chi2 csv
    indices = np.argsort(scores)[::-1]
    data2save = []
    for f in range(len(index_data)):
        data2save.append([indices[f], index_data[indices[f]], scores[indices[f]]])

    result_df_2save = pd.DataFrame(data=data2save, columns=["index", "name", "importance"])
    result_df_2save.to_csv("./result/2_chi2.csv", index=False)

    ## RF
    clf = RandomForestClassifierWithCoef(n_estimators=10000, n_jobs=-1)
    clf.fit(X, y)
    svm_weights = clf.feature_importances_ * 100
    svm_weights /= svm_weights.max()
    indices = np.argsort(svm_weights)[::-1]

    ## Plot rf
    plt.figure()
    plt.bar(X_indices - .45, svm_weights, width=.2,
            label=r'Chi2 Score (RF importance)', color='g')
    plt.savefig('./result/2_rf.png')
    plt.close()

    ## Save rf csv
    data2save = []
    for f in range(0, 100):
        data2save.append([indices[f], index_data[indices[f]], svm_weights[indices[f]]])

    result_df_2save = pd.DataFrame(data=data2save, columns=["index", "name", "importance"])
    result_df_2save.to_csv("./result/2_rf.csv", index=False)


    ## RF Selected
    clf_selected = RandomForestClassifierWithCoef(n_estimators=3000, n_jobs=-1)
    clf_selected.fit(selector.transform(X), y)
    svm_weights_selected = clf_selected.feature_importances_ * 100
    svm_weights_selected /= svm_weights_selected.max()

    indices = np.argsort(svm_weights_selected)[::-1]

    ## Plot rf
    plt.figure()
    plt.bar(np.array(X_indices)[indices] - .45, svm_weights_selected, width=.2,
            label=r'Chi2 Score (RF importance selected)', color='g')
    plt.savefig('./result/2_rf_selected.png')
    plt.close()

    ## Save rf csv
    data2save = []
    for f in range(0, 20):
        data2save.append([indices[f], index_data[indices[f]], svm_weights[indices[f]]])

    result_df_2save = pd.DataFrame(data=data2save, columns=["index", "name", "importance"])
    result_df_2save.to_csv("./result/2_rf_selected.csv", index=False)

    ## Selected the best number of feature
    # rf = RandomForestClassifier(n_estimators=2000, n_jobs=-1)
    svc = SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=100, cv=StratifiedKFold(y, 2),
              scoring='roc_auc')
    rfecv.fit(X, y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.savefig("2_rfecv.png")
    plt.show()


