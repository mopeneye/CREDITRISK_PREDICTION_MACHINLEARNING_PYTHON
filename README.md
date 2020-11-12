# CREDITRISK_PREDICTION_MACHINLEARNING_PYTHON

#
# Problem
# Establishing a machine learning model to classify credit risk.
#
# Data Set Story
#
# Data set source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
#
# There are observations of 1000 individuals.
# Variables
#
# Age: Age
#
# Sex: Gender
#
# Job: Job-Ability (0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
#
# Housing: Housing Status (own, rent, or free)
#
# Saving accounts: Saving Status (little, moderate, quite rich, rich)
#
# Checking account: Current Account (DM - Deutsch Mark)
#
# Credit amount: Credit Amount (DM)
#
# Duration: Duration (month)
#
# Purpose: Purpose (car, furniture / equipment, radio / TV, domestic appliances, repairs, education, business, vacation / others)
#
# Risk: Risk (Good, Bad Risk)

results

# LGBM
# Base:  LightGBM: 0.758000 (0.036000)
# LGBM Baslangic zamani:  2020-11-09 22:55:59.023605
# Fitting 10 folds for each of 144 candidates, totalling 1440 fits
# LGBM Bitis zamani:  2020-11-09 23:00:29.306755
# LGBM Tuned:  LightGBM: 0.766000 (0.000000)
# LGBM Best params:  {'learning_rate': 0.01, 'max_depth': 8, 'n_estimators': 500, 'num_leaves': 50}
#
# RF
# Base:  RF: 0.740000 (0.020000)
# RF Baslangic zamani:  2020-11-09 16:56:06.939579
# Fitting 10 folds for each of 60 candidates, totalling 600 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# RF Bitis zamani:  2020-11-09 17:08:02.005124
# RF Tuned:  RF: 0.759000 (0.000000)
# RF Best params:  {'max_depth': 50, 'max_features': 5, 'min_samples_split': 20, 'n_estimators': 500}
#
#
# KNN
# Base:  KNN: 0.671000 (0.036180)
# XKNN Baslangic zamani:  2020-11-09 17:17:44.633991
# Fitting 10 folds for each of 49 candidates, totalling 490 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# KNN Bitis zamani:  2020-11-09 17:17:49.196383
# KNN Tuned:  KNN: 0.731000 (0.000000)
# KNN Best params:  {'n_neighbors': 28}
#
#
# SVC
# Base:  SVC: 0.731000 (0.017000)
# SVC Baslangic zamani:  2020-11-09 17:42:24.504876
# Fitting 10 folds for each of 36 candidates, totalling 360 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# ?SVC Bitis zamani:  2020-11-09 19:38:40.977620
# SVC Tuned:  SVC: 0.751000 (0.000000)
# SVC Best params:  {'C': 4, 'kernel': 'linear'}
#
#
# XGB
# Base:  XGB: 0.770000 (0.034928)
# XGB Baslangic zamani:  2020-11-09 23:09:12.546576
# Fitting 10 folds for each of 270 candidates, totalling 2700 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# XGB Bitis zamani:  2020-11-09 23:30:16.700013
# XGB Tuned:  XGB: 0.773000 (0.000000)
# XGB Best params:  {'learning_rate': 0.005, 'loss': 'exponential', 'max_depth': 5, 'n_estimators': 1000, 'subsample': 1}

