import sys
sys.path.append("../../")
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from tc_utils.losses import multiclass_logloss

from ensembling.spooky.grid_search import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

def mnb(xtrain, ytrain, xvalid, yvalid):
    clf = MultinomialNB()
    clf.fit(xtrain, ytrain)
    predictions = clf.predict_proba(xvalid)

    print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

def mnb_gridsearch(train_x, train_y, val_x, val_y):
    nb_model = MultinomialNB()

    # Create the pipeline
    clf = pipeline.Pipeline([('nb', nb_model)])

    # parameter grid
    param_grid = {'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

    # Initialize Grid Search Model
    model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                         verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

    # Fit Grid Search Model
    model.fit(train_x, train_y)  # we can use the full data here but im only using xtrain.
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def generate_nb_features(train_x, train_y, test_x, num_classes, feature_counts=5, random_seed=42):
    train_nb_features = np.zeros((train_x.shape[0], num_classes))
    test_nb_features = np.zeros((test_x.shape[0], num_classes))

    kf = KFold(n_splits=feature_counts, shuffle=True, random_state=23 * random_seed)

    for train_index, test_index in kf.split(train_x):
        train_x_tmp, test_x_tmp = train_x[train_index], train_x[test_index]
        train_y_tmp, test_y_tmp = train_y[train_index], train_y[test_index]

        tmp_model = MultinomialNB(alpha=0.025, fit_prior=False)
        tmp_model.fit(train_x_tmp, train_y_tmp)

        tmp_train_feat = tmp_model.predict_proba(test_x_tmp)

        test_feat = tmp_model.predict_proba(test_x)

        train_nb_features[test_index] = tmp_train_feat

        test_nb_features += test_feat / feature_counts
    return train_nb_features, test_nb_features
