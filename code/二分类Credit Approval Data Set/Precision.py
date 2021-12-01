import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
from sklearn import metrics
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

# models = [logistic_model,decisiontree_model,randomforest_model,svm_model,gaussion_model,KNN_model,neutralnetwork_model]
# models_Name = ['logistic_model','decisiontree_model','randomforest_model','svm_model','gaussion_model','KNN_model','neutralnetwork_model']

models = [KNN_model]
models_Name = ['KNN_model']
#正态化
def normalization(df):
    df["A2"] = yeojohnson(df["A2"])[0]
    df["A3"] = yeojohnson(df["A3"])[0]
    df["A8"] = yeojohnson(df["A8"])[0]
    df["A11"] = yeojohnson(df["A11"])[0]
    df["A14"] = yeojohnson(df["A14"])[0]
    df["A15"] = yeojohnson(df["A15"])[0]
    return df
#标准化
def standardization(df):
    from sklearn.preprocessing import StandardScaler
    StandardScaler = StandardScaler()
    columns_to_scale = ['A2','A3','A8','A11','A14','A15']
    df[columns_to_scale] = StandardScaler.fit_transform(df[columns_to_scale])
    return df
#归一化
def preprocessing(df):
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    columns_to_scale =['A2','A3','A8','A11','A14','A15']
    df[columns_to_scale] = min_max_scaler.fit_transform(df[columns_to_scale])
    return df
#哑变量
def dummyvariable(df):
    a = pd.get_dummies(df['A1'], prefix = "A1")#get_dummies进行ont hot编码
    b = pd.get_dummies(df['A4'], prefix="A4")
    c = pd.get_dummies(df['A5'], prefix="A5")
    d = pd.get_dummies(df['A6'], prefix="A6")
    e = pd.get_dummies(df['A7'], prefix="A7")
    f = pd.get_dummies(df['A9'], prefix="A9")
    g = pd.get_dummies(df['A10'], prefix="A10")
    h = pd.get_dummies(df['A12'], prefix="A12")
    i = pd.get_dummies(df['A13'], prefix="A13")
    frames = [df, a, b, c, d, e,f,g,h,i]
    df = pd.concat(frames, axis=1)
    df = df.drop(columns=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])
    return df


for i in range(len(models)):
    #无处理
    df = pd.read_csv("crx.csv")
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化
    df = pd.read_csv("crx.csv")
    df=normalization(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化
    df = pd.read_csv("crx.csv")
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化
    df = pd.read_csv("crx.csv")
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化
    df = pd.read_csv("crx.csv")
    df = normalization(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化
    df = pd.read_csv("crx.csv")
    df = normalization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化--归一化
    df = pd.read_csv("crx.csv")
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化--标准化
    df = pd.read_csv("crx.csv")
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化--归一化
    df = pd.read_csv("crx.csv")
    df = normalization(df)
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化--标准化
    df = pd.read_csv("crx.csv")
    df = normalization(df)
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')


    #无处理  (哑变量)
    df = pd.read_csv("crx.csv")
    df = dummyvariable(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')
    #正态化  (哑变量)
    df = pd.read_csv("crx.csv")
    df = dummyvariable(df)
    df=normalization(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化  (哑变量)
    df = pd.read_csv("crx.csv")
    df = dummyvariable(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化  (哑变量)
    df = pd.read_csv("crx.csv")
    df = dummyvariable(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化  (哑变量)
    df = pd.read_csv("crx.csv")
    df = dummyvariable(df)
    df = normalization(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化  (哑变量)
    df = pd.read_csv("crx.csv")
    df = dummyvariable(df)
    df = normalization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化--归一化  (哑变量)
    df = pd.read_csv("crx.csv")
    df = dummyvariable(df)
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化--标准化  (哑变量)
    df = pd.read_csv("crx.csv")
    df = dummyvariable(df)
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化--归一化  (哑变量)
    df = pd.read_csv("crx.csv")
    df = dummyvariable(df)
    df = normalization(df)
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化--标准化  (哑变量)
    df = pd.read_csv("crx.csv")
    df = dummyvariable(df)
    df = normalization(df)
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['A16'], axis=1)
    y = df['A16']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')
