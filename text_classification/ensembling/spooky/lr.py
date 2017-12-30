import sys

from sklearn.decomposition import TruncatedSVD

sys.path.append("../")
sys.path.append("../../")
from sklearn.linear_model import LogisticRegression
from tc_utils.losses import multiclass_logloss
from ensembling.spooky.grid_search import *

def lr(xtrain_tfv, ytrain, xvalid_tfv, yvalid):
    # Fitting a simple Logistic Regression on TFIDF
    clf = LogisticRegression(C=1.0)
    clf.fit(xtrain_tfv, ytrain)
    predictions = clf.predict_proba(xvalid_tfv)

    print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


def lr_svd_gridsearch(train_x, train_y, val_x, val_y):
    # Initialize SVD
    svd = TruncatedSVD()

    # Initialize the standard scaler
    scl = preprocessing.StandardScaler()

    # We will use logistic regression here..
    lr_model = LogisticRegression()

    # Create the pipeline
    clf = pipeline.Pipeline([('svd', svd),
                             ('scl', scl),
                             ('lr', lr_model)])

    param_grid = {'svd__n_components': [120, 180],
                  'lr__C': [0.1, 1.0, 10],
                  'lr__penalty': ['l1', 'l2']}

    # Initialize Grid Search Model
    model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                         verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

    # Fit Grid Search Model
    model.fit(train_x, train_y)  # we can use the full data here but im only using xtrain
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


