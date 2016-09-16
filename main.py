import numpy as np

from prepare_data import PrepareData
from sklearn.ensemble import RandomForestClassifier

prepared_data = PrepareData()

labels = prepared_data.tag
raw_data = prepared_data.raw_data
training_data = raw_data[1:]
index_data = raw_data[0]

forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=1)
forest.fit(data, labels)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(len(index_data)):
    print("%2d) %-*s %f" % (f+1, 30, index_data[f], importances[indices[f]]))
