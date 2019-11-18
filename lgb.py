import sys
import csv
import operator
import pandas as pd
import numpy as np
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
test_1 = "data/test.csv"
train_target="data/train_y.csv"
target_variable=["Loan_Status"]
columns_categorical=[]

train_X=pd.read_csv(train_path)
train_X.set_index('Loan_ID')
#train_X,cat_cols,num_cols=convertTOCategorical(train_X)
test_X=pd.read_csv(test_path)
test_X.set_index('Loan_ID')

test_1=pd.read_csv(test_1)
train_y=pd.read_csv(train_target)

train_y=handleCategoricalValues(train_y, target_variable)
#train_X = train_X.drop(columns=target_variable, axis=1)
train_X = train_X.drop(columns=['Loan_ID'], axis=1)
target_ref=pd.DataFrame()
target_ref['Loan_ID'] = test_1['Loan_ID']
test_X = test_X.drop(columns=['Loan_ID'], axis=1)
test_X
# scaler = preprocessing.StandardScaler()
# scaler.fit(train_X)
# train_X = scaler.transform(train_X)
# test_X =  scaler.transform(test_X)

def runLGB(train_X, train_y, test_X, test_y=None, test_X2=None, dep=8, seed=0, rounds=2500):
    params = {}
    params["objective"] = "binary"
    params['metric'] = 'roc'#'binary_logloss'
    params["max_depth"] = dep
    params["min_data_in_leaf"] = 200
    params["learning_rate"] = 0.01
    params["bagging_fraction"] = 0.8
    params["feature_fraction"] = 0.35
    params["bagging_freq"] = 1
    params["bagging_seed"] = seed
    params["lambda_l2"] = 5
    params["lambda_l1"] = 5
    params["num_leaves"] = 30
    # params["reg_alpha"] = 3.4492
    # params["reg_lambda"] = 0.0422
    # params["n_estimators"] = 197
    #params["verbosity"] = -1
    num_rounds = rounds
    plst = list(params.items())
    lgtrain = lgb.Dataset(train_X, label=train_y)
    if test_y is not None:
        lgtest = lgb.Dataset(test_X, label=test_y)
        model = lgb.train(params, lgtrain, num_rounds, valid_sets=[lgtest], early_stopping_rounds=200, verbose_eval=1000)
    else:
        lgtest = lgb.Dataset(test_X)
        model = lgb.train(params, lgtrain, num_rounds)
        pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    if test_X2 is not None:
        pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
        imps = model.feature_importance()
        names = model.feature_name()
        for fi, fn in enumerate(names):
            print(fn, imps[fi])
    loss = 0
    if test_y is not None:
        loss = np.sqrt(metrics.mean_squared_error(test_y, pred_test_y))
        print(loss)
        return pred_test_y, loss, pred_test_y2, model.best_iteration
    else:
        return pred_test_y
    
sub_df = pd.DataFrame()
preds = runLGB(train_X, train_y, test_X, seed=0)
sub_df["Loan_ID"] = target_ref['Loan_ID']
sub_df["Loan_Status"] = preds
type(sub_df["Loan_Status"])

condlist = [
   sub_df["Loan_Status"] >= 0.65,
   sub_df["Loan_Status"] < 0.65
]
choicelist = ['Y', 'N']
sub_df["Loan_Status"] = np.select(
   condlist, choicelist, default='Y')

sub_df[sub_df["Loan_Status"]=='Y'].count()
sub_df.to_csv("data/result_lgb.csv", index=False)
