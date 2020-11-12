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

# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import LocalOutlierFactor

import warnings
warnings.filterwarnings("ignore")

def load_CreditRisk_data():
    df = pd.read_csv(r'E:\PROJECTS\dsmlbc\CreditRiskPredictiıon\datasets\credit_risk.csv', index_col=0)
    return df

df = load_CreditRisk_data()

# EDA

# OVERVIEW

print(df.head())

print(df.tail())

print(df.info())

print(df.columns)

print(df.shape)

print(df.index)

print(df.describe().T)

print(df.isnull().values.any())

print(df.isnull().sum().sort_values(ascending=False))

# INDEPENDENT VARIABLE OPERATIONS

# convert JOB column to categorical
category = pd.cut(df.Job, bins=[-1, 0.9, 1.9, 2.9, 3.9],labels=['unskilled_nonres', 'unskilled_res', 'skilled', 'highly_skilled'])
df.insert(2, 'Job_category', category)
df['Job_category'] = df['Job_category'].astype('O')
df.drop('Job', axis=1, inplace=True)

# Customers have saving, but don't have Checking account
df[(df['Saving accounts'].isnull() == 0) & (df['Checking account'].isnull())] [['Saving accounts', 'Checking account']]

# Convert Checking account nulls to None
df.loc[(df['Checking account'].isnull()), 'Checking account'] = 'None'

# Fill Saving accounmts Null values with Checking account
df.loc[(df['Saving accounts'].isnull()), 'Saving accounts'] = df.loc[(df['Saving accounts'].isnull())]['Checking account']

# Convert Duration column to category
# category2 = pd.cut(df.Duration, bins=[0, 11, 21, 31, 80],labels=['short', 'medium', 'high', 'veryhigh'])
# df.insert(7, 'Duration_category', category2)
# df['Duration_category'] = df['Duration_category'].astype('O')
# df.drop('Duration', axis=1, inplace=True)

# Convert Risk to 1 and 0
df['Risk'] = df['Risk'].replace('bad',1)
df['Risk'] = df['Risk'].replace('good',0)

# Create Has_Money column
df.loc[(df['Saving accounts'] != 'None') | (df['Checking account'] != 'None'), 'Has_Money'] = 1
df['Has_Money'] = df['Has_Money'].replace(np.nan, 0).astype(('int'))

# NEW FEATURES RELATED WITH AGE AND SEX
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'


cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
print('Categorical Variable count: ', len(cat_cols))
print(cat_cols)

# HOW MANY CLASSES DO CATEGORICAL VARIABLES HAVE?

print(df[cat_cols].nunique())

def cats_summary(data, categorical_cols, number_of_classes=10):
    var_count = 0  # count of categorical variables will be reported
    vars_more_classes = []  # categorical variables that have more than a number specified.
    for var in data:
        if var in categorical_cols:
            if len(list(data[var].unique())) <= number_of_classes:  # choose according to class count
                print(pd.DataFrame({var: data[var].value_counts(), "Ratio": 100 * data[var].value_counts() / len(data)}), end="\n\n\n")
                sns.countplot(x=var, data=data)
                plt.show()
                var_count += 1
            else:
                vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)

cats_summary(df, cat_cols)

# NUMERICAL VARIABLE ANALYSIS

print(df.describe().T)

# NUMERICAL VARIABLES COUNT OF DATASET?

num_cols = [col for col in df.columns if df[col].dtypes != 'O']
print('Numerical Variables Count: ', len(num_cols))
print('Numerical Variables: ', num_cols)

# Histograms for numerical variables?

def hist_for_nums(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")

hist_for_nums(df, num_cols)

# DISTRIBUTION OF "Risk" VARIABLE

print(df["Risk"].value_counts()) #inbalancing problem!

def plot_categories(df, cat, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row=row, col=col)
    facet.map(sns.barplot, cat, target);
    facet.add_legend()

# TARGET ANALYSIS BASED ON CATEGORICAL VARIABLES

def target_summary_with_cat(data, target):
    cats_names = [col for col in data.columns if len(data[col].unique()) < 10 and col not in target]
    for var in cats_names:
        print(pd.DataFrame({"TARGET_MEAN": data.groupby(var)[target].mean()}), end="\n\n\n")
        plot_categories(df, cat=var, target='Risk')
        plt.show()

target_summary_with_cat(df, "Risk")

# TARGET ANALYSIS BASED ON NUMERICAL VARIABLES

def target_summary_with_nums(data, target):
    num_names = [col for col in data.columns if len(data[col].unique()) > 5
                 and df[col].dtypes != 'O'
                 and col not in target]

    for var in num_names:
        print(df.groupby(target).agg({var: np.mean}), end="\n\n\n")

target_summary_with_nums(df, "Risk")

# INVESTIGATION OF NUMERICAL VARIABLES EACH OTHER

def correlation_matrix(df):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[num_cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w',
                      cmap='RdBu')
    plt.show()

correlation_matrix(df)

# 6. WORK WITH OUTLIERS

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 0.5 * interquantile_range
    return low_limit, up_limit


num_cols2 = [col for col in df.columns if df[col].dtypes != 'O' and len(df[col].unique()) > 10]

def Has_outliers(data, number_col_names, plot=False):
    Outlier_variable_list = []

    for col in number_col_names:
        low, high = outlier_thresholds(df, col)

        if (df[(data[col] < low) | (data[col] > high)].shape[0] > 0):
            Outlier_variable_list.append(col)
            if (plot == True):
                sns.boxplot(x=col, data=df)
                plt.show()
    print('Variables that has outliers: ', Outlier_variable_list)
    return Outlier_variable_list


# def Replace_with_thresholds(data, col):
#     low, up = outlier_thresholds(data, col)
#     data.loc[(data[col] < low), col] = low
#     data.loc[(data[col] > up), col] = up
#     print("Outliers for ", col, "column have been replaced with thresholds ",
#           low, " and ", up)
#
#
# var_names = Has_outliers(df, num_cols2, True)
#
# # print(var_names)
#
# for col in var_names:
#     Replace_with_thresholds(df, col)



# MISSING VALUE ANALYSIS

# Is there any missing values
print(df.isnull().values.any()) #NO!

# 8. LABEL ENCODING

def label_encoder(dataframe):
    labelencoder = preprocessing.LabelEncoder()

    label_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"
                  and len(dataframe[col].value_counts()) == 2]

    for col in label_cols:
        dataframe[col] = labelencoder.fit_transform(dataframe[col])
    return dataframe

#
df = label_encoder(df)

# ONE-HOT ENCODING
def one_hot_encoder(dataframe, category_freq=20, nan_as_category=False):
    categorical_cols = [col for col in dataframe.columns if len(dataframe[col].value_counts()) < category_freq
                        and dataframe[col].dtypes == 'O']

    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)

    return dataframe


df = one_hot_encoder(df)

#LOF applied

clf = LocalOutlierFactor(n_neighbors = 20, contamination=0.1)

clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_

# np.sort(df_scores)[0:1000]

threshold = np.sort(df_scores)[100]

outlier_tbl = df_scores > threshold

press_value = df[df_scores == threshold]
outliers = df[~outlier_tbl]

res = outliers.to_records(index = False)
res[:] = press_value.to_records(index = False)

df[~outlier_tbl] = pd.DataFrame(res, index = df[~outlier_tbl].index)

Has_outliers(df, num_cols2, True)


# Drop Unimportant columns

# LGBM --> ['Job_category_unskilled_nonres',
#         'Saving accounts_quite rich',
#        'Purpose_domestic appliances'
#        'Purpose_repairs',
#        'Purpose_vacation/others',
#        'NEW_SEX_CAT_seniorfemale',
#        'NEW_SEX_CAT_youngfemale'],
#       dtype='object')

# df.drop(['Job_category_unskilled_nonres',
#         'Saving accounts_quite rich',
#        'Purpose_domestic appliances',
#        'Purpose_repairs',
#        'Purpose_vacation/others',
#        'NEW_SEX_CAT_seniorfemale', 'NEW_SEX_CAT_youngfemale'], axis=1, inplace=True)
# RF -->

# MODELLING

y = df["Risk"]
X = df.drop(["Risk"], axis=1)

models = [#('RF', RandomForestClassifier())]
#      ('XGB', GradientBoostingClassifier())]
#          ("LightGBM", LGBMClassifier())]
        # ('KNN', KNeighborsClassifier())]
            ('SVC', SVC())]

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=123)
    cv_results = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print('Base: ', msg)

    # RF Tuned
    if name == 'RF':
        rf_params = {"n_estimators": [500, 1000, 1500],
                     "max_features": [5, 10],
                     "min_samples_split": [20, 50],
                     "max_depth": [50, 100, None]}

        rf_model = RandomForestClassifier(random_state=123)
        print('RF Baslangic zamani: ', datetime.now())
        gs_cv = GridSearchCV(rf_model,
                             rf_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X, y)
        rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, verbose=2, n_jobs=-1).fit(X, y)  # ???
        print('RF Bitis zamani: ', datetime.now())
        rf_tuned = RandomForestClassifier(**gs_cv.best_params_).fit(X, y)
        cv_results = cross_val_score(rf_tuned, X, y, cv=10, scoring="accuracy").mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print('RF Tuned: ', msg)
        print('RF Best params: ', gs_cv.best_params_)

        # Feature Importance
        feature_imp = pd.Series(rf_tuned.feature_importances_,
                                index=X.columns).sort_values(ascending=False)

        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Değişken Önem Skorları')
        plt.ylabel('Değişkenler')
        plt.title("Değişken Önem Düzeyleri")
        plt.show()
        plt.savefig('rf_importances.png')

    # LGBM Tuned
    elif name == 'LightGBM':
        lgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
                       "n_estimators": [500, 750, 1000],
                       "max_depth": [8, 15, 20, 30],
                       'num_leaves': [31, 50, 100, 200]}

        lgbm_model = LGBMClassifier(random_state=123)
        print('LGBM Baslangic zamani: ', datetime.now())
        gs_cv = GridSearchCV(lgbm_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X, y)
        print('LGBM Bitis zamani: ', datetime.now())
        lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X, y)
        cv_results = cross_val_score(lgbm_tuned, X, y, cv=10, scoring="accuracy").mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print('LGBM Tuned: ', msg)
        print('LGBM Best params: ', gs_cv.best_params_)

        # Feature Importance
        feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                                index=X.columns).sort_values(ascending=False)

        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Değişken Önem Skorları')
        plt.ylabel('Değişkenler')
        plt.title("Değişken Önem Düzeyleri")
        plt.show()
        plt.savefig('lgbm_importances.png')

    # XGB Tuned
    elif name == 'XGB':
        xgb_params = {  # "colsample_bytree": [0.05, 0.1, 0.5, 1],
            'max_depth': np.arange(1, 6),
            'subsample': [0.5, 0.75, 1],
            'learning_rate': [0.005, 0.01, 0.05],
            'n_estimators': [500, 1000, 1500],
            'loss': ['deviance', 'exponential']}

        xgb_model = GradientBoostingClassifier(random_state=123)

        print('XGB Baslangic zamani: ', datetime.now())
        gs_cv = GridSearchCV(xgb_model,
                             xgb_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X, y)
        print('XGB Bitis zamani: ', datetime.now())
        xgb_tuned = GradientBoostingClassifier(**gs_cv.best_params_).fit(X, y)
        cv_results = cross_val_score(xgb_tuned, X, y, cv=10, scoring="accuracy").mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print('XGB Tuned: ', msg)
        print('XGB Best params: ', gs_cv.best_params_)

    # KNN Tuned
    elif name == 'KNN':
        knn_params = {"n_neighbors": np.arange(1,50)}

        knn_model = KNeighborsClassifier()

        print('KNN Baslangic zamani: ', datetime.now())
        gs_cv = GridSearchCV(knn_model,
                             knn_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X, y)
        print('KNN Bitis zamani: ', datetime.now())
        knn_tuned = KNeighborsClassifier(**gs_cv.best_params_).fit(X, y)
        cv_results = cross_val_score(knn_tuned, X, y, cv=10, scoring="accuracy").mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print('KNN Tuned: ', msg)
        print('KNN Best params: ', gs_cv.best_params_)

        # SVC Tuned
    elif name == 'SVC':
        svc_params = {"C": np.arange(1,10),
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

        svc_model = SVC()

        print('SVC Baslangic zamani: ', datetime.now())
        gs_cv = GridSearchCV(svc_model,
                             svc_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X, y)
        print('SVC Bitis zamani: ', datetime.now())
        svc_tuned = SVC(**gs_cv.best_params_).fit(X, y)
        cv_results = cross_val_score(svc_tuned, X, y, cv=10, scoring="accuracy").mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print('SVC Tuned: ', msg)
        print('SVC Best params: ', gs_cv.best_params_)


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