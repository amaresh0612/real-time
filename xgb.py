# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:00:46 2019

@author: -
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:27:31 2019

@author: amareshdash
"""

import sys
import csv
import operator
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold,train_test_split
from sklearn import preprocessing, model_selection, metrics, ensemble
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from utilities import convertTOCategorical,handleCategoricalValues
from sklearn.model_selection import GridSearchCV
from loader import loader_fn

loader_fn()


train_path = "data/train_X.csv"
test_path = "data/test_X.csv"
test_1_path = "data/test.csv"
train_target="data/train_y.csv"
target_variable=["Loan_Status"]
ID = 'Loan_ID'

train_X=pd.read_csv(train_path)
train_X.set_index(ID)
test_X=pd.read_csv(test_path)
test_X.set_index(ID)

train_X.select_dtypes(exclude=["number"]).columns
test_X.select_dtypes(exclude=["number"]).columns

train_y=pd.read_csv(train_target)
test_1 = pd.read_csv(test_1_path)

train_y=handleCategoricalValues(train_y, target_variable)
train_X = train_X.drop(columns=ID, axis=1)
target_ref=pd.DataFrame()
target_ref[ID] = test_1[ID]
test_X = test_X.drop(columns=ID, axis=1)



def getCountVar(compute_df, count_df, var_name, count_var="Num"):
    grouped_df = count_df.groupby(var_name, as_index=False)[count_var].agg('count')
    grouped_df.columns = [var_name, "var_count"]
    merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
    merged_df.fillna(-1, inplace=True)
    return list(merged_df["var_count"])


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0):
    params = {}
    params["objective"] = "binary:logistic"
    params['eval_metric'] = 'auc'
    params["eta"] = 0.01  # 0.00334
    params["min_child_weight"] = 1
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.3
    params["silent"] = 1
    params["max_depth"] = 6
    params["seed"] = seed_val
    # params["max_delta_step"] = 2
    # params["gamma"] = 0.5
    num_rounds = 1000  # 2500

    plst = list(params.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=500)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    if feature_names:
        create_feature_map(feature_names)
        model.dump_model('xgbmodel.txt', 'xgb.fmap', with_stats=True)
        importance = model.get_fscore(fmap='xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
        imp_df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()
        imp_df.to_csv("imp_feat.txt", index=False)

    pred_test_y = model.predict(xgtest)

    if test_y is not None:
        loss = roc_auc_score(test_y, pred_test_y)
        print(loss)
        return pred_test_y, loss
    else:
        return pred_test_y


feat_names = list(train_X.columns)
train_X = np.array(train_X)
test_X = np.array(test_X)
train_X.shape[1] == test_X.shape[1]


preds = runXGB(train_X, train_y, test_X, feature_names=feat_names, seed_val = 0)
sub_df = pd.DataFrame()
sub_df[ID] = target_ref[ID]
sub_df["Loan_Status"] = preds

condlist = [
   sub_df["Loan_Status"] >= 0.65,
   sub_df["Loan_Status"] < 0.65
]
choicelist = ['Y', 'N']
sub_df["Loan_Status"] = np.select(condlist, choicelist, default='Y')
sub_df.to_csv("data/result_xgb.csv", index=False)
