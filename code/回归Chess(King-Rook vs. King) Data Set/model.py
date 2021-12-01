import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings

from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
warnings.filterwarnings("ignore")
from scipy.stats import boxcox
from scipy.stats import yeojohnson
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, classification_report, roc_auc_score, roc_curve, \
    confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn import svm
def SVM_model(X_train, X_test,y_train, y_test):
    lr_model = svm.SVR()

    lr_model.fit(X_train, y_train)

    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(lr_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    return result

def Linear_model(X_train, X_test,y_train, y_test):
    from sklearn import linear_model
    lr_model = linear_model.LinearRegression()

    lr_model.fit(X_train, y_train)

    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(lr_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    return result

def DTree_model(X_train, X_test,y_train, y_test):
    from sklearn import tree
    lr_model = tree.DecisionTreeRegressor()

    lr_model.fit(X_train, y_train)

    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(lr_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    return result

def KNN_model(X_train, X_test,y_train, y_test):
    from sklearn import neighbors
    lr_model = neighbors.KNeighborsRegressor()
    lr_model.fit(X_train, y_train)

    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(lr_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    return result

def RandomForest_model(X_train, X_test,y_train, y_test):
    from sklearn import ensemble
    lr_model = ensemble.RandomForestRegressor(n_estimators=50)
    lr_model.fit(X_train, y_train)

    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(lr_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    return result

def AdaBoost_model(X_train, X_test,y_train, y_test):
    from sklearn import ensemble
    lr_model = ensemble.AdaBoostRegressor(n_estimators=50)
    lr_model.fit(X_train, y_train)

    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(lr_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    return result

def GBRT_model(X_train, X_test,y_train, y_test):
    from sklearn import ensemble
    lr_model = ensemble.GradientBoostingRegressor(n_estimators=100)
    lr_model.fit(X_train, y_train)

    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(lr_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    return result

def Bagging_model(X_train, X_test,y_train, y_test):
    from sklearn.ensemble import BaggingRegressor
    lr_model = BaggingRegressor()
    lr_model.fit(X_train, y_train)

    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(lr_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    return result

def ExtraTree_model(X_train, X_test,y_train, y_test):
    from sklearn.tree import ExtraTreeRegressor
    lr_model = ExtraTreeRegressor()
    lr_model.fit(X_train, y_train)

    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(lr_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    return result

def ARD_model(X_train, X_test,y_train, y_test):
    lr_model = linear_model.ARDRegression()
    lr_model.fit(X_train, y_train)

    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(lr_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    return result