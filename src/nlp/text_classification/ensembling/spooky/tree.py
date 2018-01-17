import sys
sys.path.append("../../")

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from tc_utils.losses import multiclass_logloss

def xgbst_on_vec(train_x, train_y, val_x, val_y):
    '''
    Use this on TFIDF or CountVectorized features
    :param train_x: 
    :param train_y: 
    :param val_x: 
    :param val_y: 
    :return: 
    '''
    clf = XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                            subsample=0.8, nthread=10, learning_rate=0.1, silent=False)
    clf.fit(train_x.tocsc(), train_y)
    predictions = clf.predict_proba(val_x.tocsc())

    print("logloss: %0.3f " % multiclass_logloss(val_y, predictions))

def xgbst(train_x, train_y, val_x, val_y):
    '''
    Use this on TFIDF or CountVectorized features
    :param train_x: 
    :param train_y: 
    :param val_x: 
    :param val_y: 
    :return: 
    '''
    clf = XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                            subsample=0.8, nthread=10, learning_rate=0.1, silent=False)
    clf.fit(train_x, train_y)
    predictions = clf.predict_proba(val_x)

    print("logloss: %0.3f " % multiclass_logloss(val_y, predictions))