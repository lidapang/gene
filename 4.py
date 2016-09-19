import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pprint import pprint
from prepare_data import PrepareData
from sklearn import datasets, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest, chi2, RFECV
from sklearn_patch import RandomForestClassifierWithCoef

if __name__ == "__main__":
    prepared_data = PrepareData()

    labels = prepared_data.multi_pheno_tags
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
    plt.title("Chi2 Score Bar Image")
    plt.xlabel("Feature Number")
    plt.ylabel("Importance of each feature")

    plt.savefig('./result/4_chi2_scores.png')
    plt.close()

    plt.figure()
    plt.plot(X_indices, scores, 'ro', label=r'Chi2 Score (scores)')
    plt.title("Chi2 Score Point Image")
    plt.xlabel("Feature Number")
    plt.ylabel("Importance of each feature")

    plt.savefig('./result/4_chi2_scores_point.png')
    plt.close()
    ## Plot chi2 pvalues
    plt.figure()
    plt.bar(X_indices - .45, pvalues, width=.2,
            label=r'Chi2 Score ($-Log(p_{value})$)', color='g')
    plt.xlabel("Feature Number")
    plt.ylabel("Importance of each feature")

    plt.savefig('./result/4_chi2_pvalues.png')
    plt.close()
    ## Save chi2 csv
    indices = np.argsort(scores)[::-1]
    data2save = []
    for f in range(len(index_data)):
        data2save.append([indices[f], index_data[indices[f]], scores[indices[f]]])

    result_df_2save = pd.DataFrame(data=data2save, columns=["index", "name", "importance"])
    result_df_2save.to_csv("./result/4_chi2.csv", index=False)

    ## RF
    clf = RandomForestClassifierWithCoef(n_estimators=10000, n_jobs=-1)
    clf.fit(X, y)
    svm_weights = clf.feature_importances_ * 100
    svm_weights /= svm_weights.max()
    indices = np.argsort(svm_weights)[::-1]

    ## Plot rf
    plt.figure()
    plt.bar(X_indices - .45, svm_weights, width=.2,
            label=r'RF importance', color='g')
    plt.title("Random Forest Importance Bar Image")
    plt.xlabel("Feature Number")
    plt.ylabel("Importance of each feature")

    plt.savefig('./result/4_rf.png')
    plt.close()

    plt.figure()
    plt.plot(X_indices, svm_weights, 'ro', label=r'RF importance')
    plt.title("Random Forest Importance Point Image")
    plt.xlabel("Feature Number")
    plt.ylabel("Importance of each feature")

    plt.savefig('./result/4_rf_point.png')
    plt.close()
    ## Save rf csv
    data2save = []
    for f in range(0, 100):
        data2save.append([indices[f], index_data[indices[f]], svm_weights[indices[f]]])

    result_df_2save = pd.DataFrame(data=data2save, columns=["index", "name", "importance"])
    result_df_2save.to_csv("./result/4_rf.csv", index=False)

    ## Selected the best number of feature
    # rf = RandomForestClassifier(n_estimators=2000, n_jobs=-1)
    # rf = RandomForestClassifier(n_estimators=5000, n_jobs=-1)
    # svc = SVC(kernel="linear")
    # lr = LogisticRegression(penalty='l2', n_jobs=-1)
    # rfecv = RFECV(estimator=rf, step=50, cv=StratifiedKFold(y, 5),
    #           scoring='roc_auc')
    # rfecv.fit(X, y)
    # print("Optimal number of features : %d" % rfecv.n_features_)
    # plt.figure()
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (nb of correct classifications)")
    # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    # plt.savefig("4_rfecv.png")
    # plt.show()



