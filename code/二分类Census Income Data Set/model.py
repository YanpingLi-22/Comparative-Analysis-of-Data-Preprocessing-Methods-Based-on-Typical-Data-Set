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
    lr = LogisticRegression(tol=5, random_state=999)
    # Setting parameters for GridSearchCV
    params = {'penalty': ['l1', 'l2'],
              'C': [0.01, 0.1, 1, 10, 100],
              'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
              'class_weight': ['balanced', None]}
    lr_model = GridSearchCV(lr, param_grid=params)
    lr_model.fit(X_train, y_train)
    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(lr_model.best_estimator_.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    y_score = lr_model.predict_proba(X_test)[:,1]
    return result,y_score

def decisiontree_model(X_train, X_test,y_train, y_test):
    dtree = DecisionTreeClassifier(random_state=999)
    # Setting parameters for GridSearchCV
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
    y_score = tree_model.predict_proba(X_test)[:,1]
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
    y_score = rf_model.predict_proba(X_test)[:,1]
    return result,y_score

def svm_model(X_train, X_test,y_train, y_test):
    svm_model = SVC(probability=True)
    svm_model.fit(X_train, y_train)
    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(svm_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    y_score = svm_model.predict_proba(X_test)[:,1]
    return result,y_score

def gaussion_model(X_train, X_test,y_train, y_test):
    NB = GaussianNB()
    NB_model = NB.fit(X_train, y_train)

    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(NB_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    y_score = NB_model.predict_proba(X_test)[:,1]
    return result,y_score
def KNN_model(X_train, X_test,y_train, y_test):
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    real = pd.DataFrame(y_test.tolist())
    pred = pd.DataFrame(knn_model.predict(X_test).tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real
    y_score = knn_model.predict_proba(X_test)[:,1]
    return result,y_score


class Net(nn.Module):
    def __init__(self, in_count, output_count):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_count, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, output_count)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return self.softmax(x)

def neutralnetwork_model(X_train, X_test,y_train, y_test):
    X_train1 = torch.FloatTensor(np.array(X_train))
    X_test1 = torch.FloatTensor(np.array(X_test))
    y_train1 = torch.LongTensor(np.array(y_train))
    y_test1 = torch.LongTensor(np.array(y_test))
    model = Net(X_train1.shape[1], 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    final_losses = []
    for epochs in range(1500):
        optimizer.zero_grad()
        out = model(X_train1)
        loss = criterion(out, y_train1)
        final_losses.append(loss)
        loss.backward()
        optimizer.step()

    pred = model(X_test1)
    _, predict_classes = torch.max(pred, 1)
    real = pd.DataFrame(y_test1.tolist())
    pred = pd.DataFrame(predict_classes.tolist())
    real['pred'] = pred
    real.columns = ['real', 'pred']
    result = real

    return result,pred

