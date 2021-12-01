import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def logistic_model(X_train, X_test,y_train, y_test):
    lr_model = LogisticRegression(tol=5, random_state=999)
    lr_model.fit(X_train, y_train)


    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(lr_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    y_score = lr_model.predict_proba(X_test)
    return result,y_score

def decisiontree_model(X_train, X_test,y_train, y_test):#normalized-losses num-of-doors bore stroke horsepower peak-rpm price
    dtree = DecisionTreeClassifier(random_state=999)
    params = {'criterion': ['gini', 'entropy'],
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth': [*range(1, 10)],
              'min_samples_leaf': [*range(1, 50, 5)]}
    tree_model = GridSearchCV(dtree, param_grid=params)
    tree_model.fit(X_train, y_train)
    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(tree_model.best_estimator_.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    y_score = tree_model.predict_proba(X_test)
    return result,y_score

def randomforest_model(X_train, X_test,y_train, y_test):
    rf = RandomForestClassifier(random_state=999)
    params = {'max_depth': [*range(1, 20, 2)],
              'n_estimators': [*range(1, 200, 50)]}
    rf_model = GridSearchCV(rf, params, cv=3)
    rf_model.fit(X_train, y_train)
    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(rf_model.best_estimator_.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    y_score = rf_model.predict_proba(X_test)
    return result,y_score

def svm_model(X_train, X_test,y_train, y_test):
    svm = SVC(probability=True)
    params = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100]}
    svm_model = GridSearchCV(svm, params, cv=3)
    svm_model.fit(X_train, y_train)
    print(svm_model.best_params_)
    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(svm_model.best_estimator_.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    y_score = svm_model.predict_proba(X_test)
    return result,y_score

def gaussion_model(X_train, X_test,y_train, y_test):
    NB = GaussianNB()
    NB_model = NB.fit(X_train, y_train)

    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(NB_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    y_score = NB_model.predict_proba(X_test)
    return result,y_score
def KNN_model(X_train, X_test,y_train, y_test):
    knn = KNeighborsClassifier()
    params = {'n_neighbors': list(range(1, 20)),
              'p': list(range(1, 10)),
              'weights': ['uniform', 'distance']}
    knn_model = GridSearchCV(knn, params, cv=3)
    knn_model.fit(X_train, y_train)
    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(knn_model.best_estimator_.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    y_score = knn_model.predict_proba(X_test)
    return result,y_score


def neutralnetwork_model(X_train, X_test,y_train, y_test):
    X_train1 = torch.FloatTensor(np.array(X_train))
    X_test1 = torch.FloatTensor(np.array(X_test))
    y_train1 = torch.LongTensor(np.array(y_train))
    y_test1 = torch.LongTensor(np.array(y_test))

    net1 = torch.nn.Sequential(
        nn.Linear(X_train1.shape[1], 100),
        torch.nn.ReLU(),
        nn.Linear(100, 50),
        torch.nn.ReLU(),
        nn.Linear(50, 4)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net1.parameters(), lr=0.01)

    for t in range(2000):
        prediction = net1(X_train1)

        loss = criterion(prediction, y_train1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pred = net1(X_test1)
    _, predict_classes = torch.max(pred, 1)
    real = pd.DataFrame(y_test1.tolist())
    pred = pd.DataFrame(predict_classes.tolist())
    print(pred)
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    return result,pred

