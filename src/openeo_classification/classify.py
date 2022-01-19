import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix,plot_confusion_matrix
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("resources/training_data/final_features.csv",index_col=0)

band_names = ["B06", "B12"] + ["NDVI", "NDMI", "NDGI", "ANIR", "NDRE1", "NDRE2", "NDRE5"] + ["ratio", "VV", "VH"]
tstep_labels = ["t" + str(4 * index) for index in range(0, 6)]
all_bands = [band + "_" + stat for band in band_names for stat in ["p10", "p50", "p90", "sd"] + tstep_labels]

X = df[all_bands]
y = df["ids"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

param_grid = {'learning_rate': [0.07],#[0.03, 0.1],
        'depth': [6],#[4, 6, 10]
        'l2_leaf_reg': [3,5],#[1, 3, 5,],
        'iterations': [1500]}#, 100, 150]}
cb = CatBoostClassifier()
grid_search = GridSearchCV(estimator = cb, param_grid = param_grid, cv = 3, n_jobs = -1)

grid_search.fit(X_train, y_train)
grid_search.best_params_

gs_results = pd.DataFrame(grid_search.cv_results_).sort_values(by=["mean_test_score"],axis=0,ascending=False)
print(gs_results.head())

y_pred = grid_search.predict(X_test)
print(y_pred[0:10])

print("Accuracy on test set: "+str(accuracy_score(y_test,y_pred))[0:5])
prec, rec, fscore, sup = precision_recall_fscore_support(y_test,y_pred)
plot_confusion_matrix(grid_search,X_test, y_test)
plt.show()
