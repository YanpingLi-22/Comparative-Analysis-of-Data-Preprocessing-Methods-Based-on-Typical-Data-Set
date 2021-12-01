import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings

from model import *
from evaluate import *
warnings.filterwarnings("ignore")
from scipy.stats import boxcox
from scipy.stats import yeojohnson
from sklearn.metrics import recall_score, precision_score, classification_report, roc_auc_score, roc_curve, \
    confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

models = [SVM_model,Linear_model,DTree_model,KNN_model,RandomForest_model,AdaBoost_model,GBRT_model,Bagging_model,ExtraTree_model,ARD_model]
models_Name = ['SVM_model','Linear_model','DTree_model','KNN_model','RandomForest_model','AdaBoost_model','GBRT_model','Bagging_model','ExtraTree_model','ARD_model']

df = pd.read_csv("krkopt-new.csv")

qualvariables=['WKf','WRf','BKf']
quanvariables=['WKr','WRr','BKr']
skewnesslist=[]
kurtosislist=[]
minandmax=[]
maxvalue=[]

for i in quanvariables:
    skewnesslist.append(df[i].skew())
    kurtosislist.append(df[i].kurt())
    minandmax.append(abs(max(df[i])-min(df[i])))
    maxvalue.append(max(df[i]))


nomofskewness=np.std(skewnesslist)
nomofkurtosis=np.std(kurtosislist)

if (max(minandmax) / min(minandmax) >= 3):
    print("数据要标准化处理")
if (max(maxvalue) / min(maxvalue) >= 3):
    print("数据要归一化处理")

for i in quanvariables:
    if(abs(df[i].skew()/nomofskewness)<=1.96 and abs(df[i].kurt()/nomofkurtosis)<=1.96):
        print(i+"可以正态化")

#正态化
def normalization(df):
    df["WKr"] = yeojohnson(df["WKr"])[0]
    df["WRr"] = yeojohnson(df["WRr"])[0]
    df["BKr"] = yeojohnson(df["BKr"])[0]
    return df
#标准化
def standardization(df):
    from sklearn.preprocessing import StandardScaler
    StandardScaler = StandardScaler()
    columns_to_scale = ['WKr','WRr','BKr']
    df[columns_to_scale] = StandardScaler.fit_transform(df[columns_to_scale])
    return df
#归一化
def preprocessing(df):
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    columns_to_scale =  ['WKr','WRr','BKr']
    df[columns_to_scale] = min_max_scaler.fit_transform(df[columns_to_scale])
    return df
#哑变量
def dummyvariable(df):
    a = pd.get_dummies(df['WKf'], prefix = "WKf")#get_dummies进行ont hot编码
    b = pd.get_dummies(df['WRf'], prefix="WRf")
    c = pd.get_dummies(df['BKf'], prefix="BKf")
    frames = [df, a, b, c]
    df = pd.concat(frames, axis=1)
    df = df.drop(columns=['WKf','WRf','BKf'])
    return df


for i in range(len(models)):
       df = pd.read_csv("krkopt-new.csv")
       if(models_Name[i]=="Linear_model"):
           df = dummyvariable(df)
           df = normalization(df)
           df = preprocessing(df)
           base_model = models[i]
           X = df.drop(['WIN'], axis=1)
           y = df['WIN']
           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
           result = base_model(X_train, X_test, y_train, y_test)
           eva = evaluate(result, models_Name[i])
           Re = eva.show_all()
           Re.to_csv('1.txt', mode='a', float_format='%.4f')
       elif(models_Name[i]=="DTree_model"):
           df = normalization(df)
           df = preprocessing(df)
           base_model = models[i]
           X = df.drop(['WIN'], axis=1)
           y = df['WIN']
           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
           result = base_model(X_train, X_test, y_train, y_test)
           eva = evaluate(result, models_Name[i])
           Re = eva.show_all()
           Re.to_csv('1.txt', mode='a', float_format='%.4f')
       elif(models_Name[i]=="RandomForest_model"):
           df = normalization(df)
           df = preprocessing(df)
           base_model = models[i]
           X = df.drop(['WIN'], axis=1)
           y = df['WIN']
           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
           result = base_model(X_train, X_test, y_train, y_test)
           eva = evaluate(result, models_Name[i])
           Re = eva.show_all()
           Re.to_csv('1.txt', mode='a', float_format='%.4f')
       elif(models_Name[i]=="SVM_model"):
           df = dummyvariable(df)
           df = normalization(df)
           df = preprocessing(df)
           base_model = models[i]
           X = df.drop(['WIN'], axis=1)
           y = df['WIN']
           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
           result = base_model(X_train, X_test, y_train, y_test)
           eva = evaluate(result, models_Name[i])
           Re = eva.show_all()
           Re.to_csv('1.txt', mode='a', float_format='%.4f')
       elif (models_Name[i] == "KNN_model"):
           df = dummyvariable(df)
           df = normalization(df)
           df = preprocessing(df)
           base_model = models[i]
           X = df.drop(['WIN'], axis=1)
           y = df['WIN']
           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
           result = base_model(X_train, X_test, y_train, y_test)
           eva = evaluate(result, models_Name[i])
           Re = eva.show_all()
           Re.to_csv('1.txt', mode='a', float_format='%.4f')
       else:
           df = dummyvariable(df)
           df = normalization(df)
           df = preprocessing(df)
           base_model = models[i]
           X = df.drop(['WIN'], axis=1)
           y = df['WIN']
           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
           result = base_model(X_train, X_test, y_train, y_test)
           eva = evaluate(result, models_Name[i])
           Re = eva.show_all()
           Re.to_csv('1.txt', mode='a', float_format='%.4f')

