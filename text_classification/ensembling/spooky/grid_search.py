import sys
sys.path.append("../../")

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV

from tc_utils.losses import multiclass_logloss

mll_scorer = metrics.make_scorer(multiclass_logloss,
                                 greater_is_better=False,
                                 needs_proba=True)


