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
        temp['conf'] = 2 * temp['pred'] + temp['true']
        conf = temp.groupby('conf').count()

        # confusion matrix
        # TP FP
        # FN TN
        # TP 3; TN 0; FP 2; FN 1
        conf_mat = np.zeros(4)
        for i in range(4):
            try:
                conf_mat[i] = conf.loc[3 - i][1]
            except:
                continue

        return conf_mat.reshape(2, 2).astype(np.int16)

    def accuracy(self):
        return np.diag(self.conf_mat).sum() / np.sum(self.conf_mat)

    def precision(self):
        return self.conf_mat[0, 0] / self.conf_mat[0, :].sum()

    def recall(self):
        return self.conf_mat[0, 0] / self.conf_mat[:, 0].sum()

    def f_score(self):
        pre = self.precision()
        rec = self.recall()
        return 2 * pre * rec / (pre + rec)

    def TPR(self):
        return  self.recall()

    def FPR(self):
        return self.conf_mat[0, 1] / self.conf_mat[:, 1].sum()

    def FPR(self):
        return self.conf_mat[0, 1] / self.conf_mat[:, 1].sum()

    def AUC(self):
        Re = roc_auc_score(self.y_true,self.y_score)
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


