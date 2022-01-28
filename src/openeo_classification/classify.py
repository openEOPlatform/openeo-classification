import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix,plot_confusion_matrix
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from explore import all_crop_codes
import math
import numpy as np

df = pd.read_csv("resources/training_data/final_features.csv",index_col=0)

## Remove NaN values
df = df[df["B06_p50"].astype(int) != 65535]

band_names = ["B06", "B12"] + ["NDVI", "NDMI", "NDGI", "ANIR", "NDRE1", "NDRE2", "NDRE5"] + ["ratio", "VV", "VH"]
tstep_labels = ["t" + str(4 * index) for index in range(0, 6)]
all_bands = [band + "_" + stat for band in band_names for stat in ["p10", "p50", "p90", "sd"] + tstep_labels]

df[all_bands] = df[all_bands].astype(int)
df[["groupID","zoneID"]] = df[["groupID","zoneID"]].astype(str)

# num // 10 ** (int(math.log(num, 10)) - 4 + 1)
df["y"] = df["ids"].apply(lambda num: all_crop_codes[num])

### TEST CASE 1: TRAIN CEREALS SEPARATELY, WITHOUT TRAINING ON GRASS SPECIFICALLY
def crop_codes_y1(num):
        crop_list = [1110, 1510, 1910, # "Winter wheat", "Winter barley", "Winter cereal", # Winter cereals
                1120, 1520, 1920, #"Spring wheat", "Spring barley", "Spring cereal", # Spring / summer cereals
                4351, 1200, 5100, 8100, #"Winter rapeseed", "Maize", "Potatoes", "Sugar beet",
                # "Grasses and other fodder crops", "Temporary grass crops", "Permanent grass crops" # Grasses : 9100, 9110, 9120
        ]
        if num in crop_list:
                return all_crop_codes[num]
        else:
                return "Other"

df["y1"] = df["ids"].apply(crop_codes_y1)


### TEST CASE 2: TRAIN CEREALS SEPARATELY, WITH TRAINING ON GRASS SPECIFICALLY
def crop_codes_y2(num):
        crop_list = [1110, 1510, 1910, # "Winter wheat", "Winter barley", "Winter cereal", # Winter cereals
                1120, 1520, 1920, #"Spring wheat", "Spring barley", "Spring cereal", # Spring / summer cereals
                4351, 1200, 5100, 8100, #"Winter rapeseed", "Maize", "Potatoes", "Sugar beet",
                9100, 9110, 9120, # "Grasses and other fodder crops", "Temporary grass crops", "Permanent grass crops" # Grasses
        ]
        if num in crop_list:
                return all_crop_codes[num]
        else:
                return "Other"

df["y2"] = df["ids"].apply(crop_codes_y2)


### TEST CASE 3: TRAIN CEREALS JOINTLY, WITHOUT TRAINING ON GRASS SPECIFICALLY
def crop_codes_y3(num):
        crop_list = [
                4351, 1200, 5100, 8100, #"Winter rapeseed", "Maize", "Potatoes", "Sugar beet",
                # 9100, 9110, 9120, # "Grasses and other fodder crops", "Temporary grass crops", "Permanent grass crops" # Grasses
        ]
        if num in crop_list:
                return all_crop_codes[num]
        elif num in [1110, 1510, 1910]: # "Winter wheat", "Winter barley", "Winter cereal", # Winter cereals
                return "Winter cereals"
        elif num in [1120, 1520, 1920]: #"Spring wheat", "Spring barley", "Spring cereal", # Spring / summer cereals
                return "Spring cereals"
        else:
                return "Other"

df["y3"] = df["ids"].apply(crop_codes_y3)


### TEST CASES
## VERSCHILLENDE GROEPERINGEN VAN AEZ STRATIFICATIE
## EENTJE ZONDER STRATIFICATIE, KIJK OOK FEATURE IMPORTANCE
# X1 = df[all_bands]

## MET STRATIFICATIE ERBIJ ALS FEATURE
# X2 = df[all_bands+["groupID"]+["zoneID"]]

## MET STRATIFICATIE ALS IN LOSSE MODELLEN
# print([col for col in df.columns if col not in all_bands]) 

## GRAS ERBIJ TRAINEN OF LOS

## CEREALS LATER GROEPEREN: MAG JE DE PROBABILITIES OPTELLEN? E.G. WINTER WHEAT HEEFT 0.1 EN WINTER BARLEY 0.2 NOU DAN IS TOTAAL 0.3 EN DIE IS T


## Model training
X = df[all_bands]
y = df["y2"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=190)

param_grid = {'learning_rate': [0.07],#[0.03, 0.1],
        'depth': [6],#[4, 6, 10]
        'l2_leaf_reg': [10],#[1, 3, 5,],
        'iterations': [5000]}#, 100, 150]}
cb = CatBoostClassifier()

## Gridsearch
grid_search = GridSearchCV(estimator = cb, param_grid = param_grid, cv = 3, n_jobs = -1)

grid_search.fit(X_train, y_train)
grid_search.best_params_

gs_results = pd.DataFrame(grid_search.cv_results_).sort_values(by=["mean_test_score"],axis=0,ascending=False)
print(gs_results.head())

# y_pred = grid_search.predict(X_test)
# print(y_pred[0:10])


## Quick testing instead of running gridsearch setup
cb = CatBoostClassifier(learning_rate=0.07, depth=4,l2_leaf_reg=3, iterations=100)
cb.fit(X_train, y_train)

y_pred = cb.predict(X_test)

## Calculate feature importance
# print(grid_search.best_estimator_.feature_importances_)
feat_imp = list(zip(all_bands,cb.get_feature_importance()))
feat_imp.sort(key = lambda row: row[1], reverse=True)
print("Feature importance: ")
print(feat_imp)

print("Accuracy on test set: "+str(accuracy_score(y_test,y_pred))[0:5])
prec, rec, fscore, sup = precision_recall_fscore_support(y_test,y_pred)
# plot_confusion_matrix(grid_search,X_test, y_test)
plot_confusion_matrix(cb,X_test,y_test,xticks_rotation=90)
plt.show()


### TEST CASES
## EENTJE ZONDER STRATIFICATIE, KIJK OOK FEATURE IMPORTANCE
## MET STRATIFICATIE ERBIJ ALS FEATURE
## MET STRATIFICATIE ALS IN LOSSE MODELLEN
## VERSCHILLENDE GROEPERINGEN VAN DE STRATIFICATIE
## GRAS ERBIJ TRAINEN OF LOS
## CEREALS LATER GROEPEREN: MAG JE DE PROBABILITIES OPTELLEN? E.G. WINTER WHEAT HEEFT 0.1 EN WINTER BARLEY 0.2 NOU DAN IS TOTAAL 0.3 EN DIE IS T
