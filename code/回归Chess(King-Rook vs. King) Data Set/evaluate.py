import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics

class evaluate:
    def __init__(self, result,name):
        self.conf_mat = self.make_conf_mat(result)
        self.y_true=result['real']
        self.y_pred = result['pred']
        self.name=name
    def make_conf_mat(self, temp):
        temp = temp.copy()
        temp.columns = ['true', 'pred']
        return temp
    def MSE(self):
        return metrics.mean_squared_error(self.y_true, self.y_pred)

    def RMSE(self):
        root_mean_squared_error = np.sqrt(sum((self.y_true - self.y_pred) ** 2) / len(self.y_true))
        return root_mean_squared_error

    def R_squared(self):
        return metrics.r2_score(self.y_true, self.y_pred)

    def MAE(self):
        return metrics.mean_absolute_error(self.y_true, self.y_pred)

    def Adjusted_R_squared(self):
        n=len(self.y_true)
        p=7
        return 1-((1-metrics.r2_score(self.y_true, self.y_pred))*(n-1))/(n-p-1)
    def show_all(self):
        metric = pd.DataFrame({
            'NAME':self.name,
            'MSE':self.MSE(),
            'RMSE': self.RMSE(),
            'R_squared': self.R_squared(),
            'Adjusted_R_squared': self.Adjusted_R_squared(),
            'MAE': self.MAE(),
            }, index=['evaluation']).T
        return metric
