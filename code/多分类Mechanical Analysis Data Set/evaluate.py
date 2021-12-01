import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score

class evaluate:
    def __init__(self, result,y_score,name):
        self.conf_mat = self.make_conf_mat(result)
        self.y_true=result['real']
        self.y_score = y_score
        self.name=name
    def make_conf_mat(self, temp):
        temp = temp.copy()
        temp.columns = ['true', 'pred']
        return temp
    def accuracy(self):
        return accuracy_score(self.conf_mat['true'],self.conf_mat['pred'])
    def precision(self):
        return precision_score(self.conf_mat['true'],self.conf_mat['pred'],average='macro')
    def recall(self):
        return recall_score(self.conf_mat['true'], self.conf_mat['pred'], average='macro')
    def f_score(self):
        return f1_score(self.conf_mat['true'], self.conf_mat['pred'], average='macro')
    def TPR(self):
        return self.recall()
    def FPR(self):
        cnf_matrix = confusion_matrix(self.conf_mat['true'], self.conf_mat['pred'])
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        FPR = FP / (FP + TN)
        return np.mean(FPR)

    def AUC(self):
        dummies = pd.get_dummies(self.y_true, columns=self.y_true.to_frame().columns)
        Re = roc_auc_score(dummies, self.y_score, multi_class='ovr')
        return Re

    def show_all(self):
        metric = pd.DataFrame({
                'type':self.name,
                'accuracy': self.accuracy(),
                'precision': self.precision(),
                'recall': self.recall(),
                'f-score': self.f_score(),
                'TPR': self.TPR(),
                'FPR':self.FPR(),
                'AUC':self.AUC(),
            }, index=['evaluation']).T
        return metric
