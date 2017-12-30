import sys
sys.path.append("../../")

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.decomposition import TruncatedSVD

from tc_utils.losses import multiclass_logloss

def svd(train_x_tfidf, val_x_tfidf, n_components):
    # Apply SVD, I chose 120 components. 120-200 components are good enough for SVM model.
    svd = decomposition.TruncatedSVD(n_components=n_components)
    svd.fit(train_x_tfidf) # TODO should we consider test data also?
    xtrain_svd = svd.transform(train_x_tfidf)
    xvalid_svd = svd.transform(val_x_tfidf)

    # Scale the data obtained from SVD. Renaming variable to reuse without scaling.
    scl = preprocessing.StandardScaler()
    scl.fit(xtrain_svd)
    train_x_svd_scl = scl.transform(xtrain_svd)
    val_x_svd_scl = scl.transform(xvalid_svd)

    return train_x_svd_scl, val_x_svd_scl


def svm(train_x_svd_scl, train_y, val_x_svd_scl, val_y):
    # Fitting a simple SVM
    clf = SVC(C=1.0, probability=True)  # since we need probabilities
    clf.fit(train_x_svd_scl, train_y)
    predictions = clf.predict_proba(val_x_svd_scl)

    print("logloss: %0.3f " % multiclass_logloss(val_y, predictions))