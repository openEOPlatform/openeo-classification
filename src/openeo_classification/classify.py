import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix,plot_confusion_matrix, ConfusionMatrixDisplay
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from explore import all_crop_codes
import math
import numpy as np
from pathlib import Path
import json

df = pd.read_csv("resources/training_data/final_features.csv",index_col=0)

print(df.head())
## Other class mag iig niet op Other cereals gaan trainen
### FF open laten : seizonaal uit elkaar trekken of klassen uit elkaar trekken (wheat VS rye VS barley etc. of spring vs. winter)

## Remove NaN values
df = df[df["B06_p50"].astype(int) != 65535]

# band_names = ["B06", "B12"] + ["NDVI", "NDMI", "NDGI", "ANIR", "NDRE1", "NDRE2", "NDRE5"] + ["ratio", "VV", "VH"]
# tstep_labels = ["t" + str(4 * index) for index in range(0, 6)]
# all_bands = [band + "_" + stat for band in band_names for stat in ["p10", "p50", "p90", "sd"] + tstep_labels]
band_names_s2 = ["B06", "B12"] + ["NDVI", "NDMI", "NDGI", "ANIR", "NDRE1", "NDRE2", "NDRE5"]
band_names_s1 = ["ratio", "VV", "VH"]
tstep_labels_s2 = ["t4","t7","t10","t13","t16","t19"]
tstep_labels_s1 = ["t2","t5","t8","t11","t14","t17"]
features_s2 = [band + "_" + stat for band in band_names_s2 for stat in ["p25", "p50", "p75", "sd"] + tstep_labels_s2]
features_s1 = [band + "_" + stat for band in band_names_s1 for stat in ["p25", "p50", "p75", "sd"] + tstep_labels_s1]
all_bands = features_s2 + features_s1


df[all_bands] = df[all_bands].astype(int)
df[["groupID","zoneID"]] = df[["groupID","zoneID"]].astype(str)

# num // 10 ** (int(math.log(num, 10)) - 4 + 1)

### TODO: Dit groeperen op iedere class die ik wil predicten + de other class
df["y"] = df["id"].apply(lambda num: all_crop_codes[num])

# ### TEST CASE 1: TRAIN CEREALS SEPARATELY, WITHOUT TRAINING ON GRASS SPECIFICALLY
# def crop_codes_y1(num):
#         crop_list = [1110, 1510, 1910, # "Winter wheat", "Winter barley", "Winter cereal", # Winter cereals
#                 1120, 1520, 1920, #"Spring wheat", "Spring barley", "Spring cereal", # Spring / summer cereals
#                 4351, 1200, 5100, 8100, #"Winter rapeseed", "Maize", "Potatoes", "Sugar beet",
#                 # "Grasses and other fodder crops", "Temporary grass crops", "Permanent grass crops" # Grasses : 9100, 9110, 9120
#         ]
#         if num in crop_list:
#                 return all_crop_codes[num]
#         else:
#                 return "Other"

# df["y1"] = df["ids"].apply(crop_codes_y1)


# ### TEST CASE 2: TRAIN CEREALS SEPARATELY, WITH TRAINING ON GRASS SPECIFICALLY
# def crop_codes_y2(num):
#         crop_list = [1110, 1510, 1910, # "Winter wheat", "Winter barley", "Winter cereal", # Winter cereals
#                 1120, 1520, 1920, #"Spring wheat", "Spring barley", "Spring cereal", # Spring / summer cereals
#                 4351, 1200, 5100, 8100, #"Winter rapeseed", "Maize", "Potatoes", "Sugar beet",
#                 9100, 9110, 9120, # "Grasses and other fodder crops", "Temporary grass crops", "Permanent grass crops" # Grasses
#         ]
#         if num in crop_list:
#                 return all_crop_codes[num]
#         else:
#                 return "Other"

# df["y2"] = df["ids"].apply(crop_codes_y2)


# ### TEST CASE 3: TRAIN CEREALS JOINTLY, WITHOUT TRAINING ON GRASS SPECIFICALLY
# def crop_codes_y3(num):
#         crop_list = [
#                 4351, 1200, 5100, 8100, #"Winter rapeseed", "Maize", "Potatoes", "Sugar beet",
#                 # 9100, 9110, 9120, # "Grasses and other fodder crops", "Temporary grass crops", "Permanent grass crops" # Grasses
#         ]
#         if num in crop_list:
#                 return all_crop_codes[num]
#         elif num in [1110, 1510, 1910]: # "Winter wheat", "Winter barley", "Winter cereal", # Winter cereals
#                 return "Winter cereals"
#         elif num in [1120, 1520, 1920]: #"Spring wheat", "Spring barley", "Spring cereal", # Spring / summer cereals
#                 return "Spring cereals"
#         else:
#                 return "Other"

# df["y3"] = df["ids"].apply(crop_codes_y3)






### TEST CASES
## VERSCHILLENDE GROEPERINGEN VAN AEZ STRATIFICATIE
## EENTJE ZONDER STRATIFICATIE, KIJK OOK FEATURE IMPORTANCE
X1 = df[all_bands]

## MET STRATIFICATIE ERBIJ ALS FEATURE
# X2 = df[all_bands+["groupID"]+["zoneID"]]

## MET STRATIFICATIE ALS IN LOSSE MODELLEN
# print([col for col in df.columns if col not in all_bands]) 

## GRAS ERBIJ TRAINEN OF LOS

## CEREALS LATER GROEPEREN: MAG JE DE PROBABILITIES OPTELLEN? E.G. WINTER WHEAT HEEFT 0.1 EN WINTER BARLEY 0.2 NOU DAN IS TOTAAL 0.3 EN DIE IS T


def train_classifier(X,y,param_grid,output_path):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=190)

        cb = CatBoostClassifier()

        grid_search = GridSearchCV(estimator = cb, param_grid = param_grid, cv = 3, n_jobs = -1)

        grid_search.fit(X_train, y_train)
        print(grid_search.best_estimator_)

        ## Writing out all grid search results
        # gs_results = pd.DataFrame(grid_search.cv_results_).sort_values(by=["mean_test_score"],axis=0,ascending=False)
        # print(gs_results.head())

        ## Quick testing instead of running gridsearch setup
        # cb = CatBoostClassifier(learning_rate=0.07, depth=4,l2_leaf_reg=3, iterations=100)
        # cb.fit(X_train, y_train)
        # y_pred = cb.predict(X_test)

        ### save het model
        ### save de final hyperparams
        ### save de feature importance
        ### save the accuracy metrics

        y_pred = grid_search.predict(X_test)

        grid_search.best_estimator_.save_model(fname=str(output_path / "model.cbm"))

        feat_imp = sorted(dict(zip(all_bands,grid_search.best_estimator_.get_feature_importance())).items(), key=lambda x: x[1], reverse=True)
        # feat_imp.sort(key = lambda row: row[1], reverse=True)
        feat_imp_df = pd.DataFrame(feat_imp)
        feat_imp_df.columns = ["band", "importance"]
        print("Most important features: ")
        print(feat_imp_df.head())
        clean_feat = feat_imp_df[feat_imp_df["importance"] != 0]
        plt.bar(clean_feat["band"],clean_feat["importance"])
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(str(output_path / "feature_importance.png"))

        acc = accuracy_score(y_test,y_pred)
        print("Accuracy on test set: "+str(acc)[0:5])
        prec, rec, fscore, sup = precision_recall_fscore_support(y_test,y_pred)
        print(prec,rec,fscore,sup)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred,xticks_rotation=90)
        plt.tight_layout()
        plt.savefig(str(output_path / "confusion_matrix.png"))

        final_res = {
                "feat_importance": feat_imp,
                "accuracy": acc,
                "precision": prec.tolist(),
                "recall": rec.tolist(),
                "fscore": fscore.tolist(),
                "support": sup.tolist()
        }
        with open(str(output_path / "final_results.json"), "w") as outfile:
            json.dump(final_res, outfile, indent = 4) 



out_path = Path.cwd() / "resources" / "model1"
out_path.mkdir(parents=True, exist_ok=True)

## Model training
X = df[all_bands]
y = df["y"]
param_grid = {'learning_rate': [0.07],#[0.03, 0.1],
        'depth': [6],#[4, 6, 10]
        'l2_leaf_reg': [10],#[1, 3, 5,],
        'iterations': [10]}#, 100, 150]}
train_classifier(X,y,param_grid,out_path)





### TEST CASES
## EENTJE ZONDER STRATIFICATIE, KIJK OOK FEATURE IMPORTANCE
## MET STRATIFICATIE ERBIJ ALS FEATURE
## MET STRATIFICATIE ALS IN LOSSE MODELLEN
## VERSCHILLENDE GROEPERINGEN VAN DE STRATIFICATIE
## GRAS ERBIJ TRAINEN OF LOS
## CEREALS LATER GROEPEREN: MAG JE DE PROBABILITIES OPTELLEN? E.G. WINTER WHEAT HEEFT 0.1 EN WINTER BARLEY 0.2 NOU DAN IS TOTAAL 0.3 EN DIE IS T
