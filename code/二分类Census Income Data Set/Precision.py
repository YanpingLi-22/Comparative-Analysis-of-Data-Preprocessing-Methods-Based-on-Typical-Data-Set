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

models = [logistic_model,decisiontree_model,randomforest_model,svm_model,gaussion_model,KNN_model]
models_Name = ['logistic_model','decisiontree_model','randomforest_model','svm_model','gaussion_model','KNN_model']
models = [neutralnetwork_model]
models_Name = ['neutralnetwork_model']
#正态化
def normalization(df):
    df["age"] = yeojohnson(df["age"])[0]
    df["fnlwgt"] = yeojohnson(df["fnlwgt"])[0]
    df["education-num"] = yeojohnson(df["education-num"])[0]
    df["hours-per-week"] = boxcox(df['hours-per-week'])[0]
    return df
#标准化
def standardization(df):
    from sklearn.preprocessing import StandardScaler
    StandardScaler = StandardScaler()
    columns_to_scale = ['age','fnlwgt','education-num','hours-per-week','capital-gain','capital-loss']
    df[columns_to_scale] = StandardScaler.fit_transform(df[columns_to_scale])
    return df
#归一化
def preprocessing(df):
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    columns_to_scale =['age','fnlwgt','education-num','hours-per-week','capital-gain','capital-loss']
    df[columns_to_scale] = min_max_scaler.fit_transform(df[columns_to_scale])
    return df
#哑变量
def dummyvariable(df):
    a = pd.get_dummies(df['workclass'], prefix = "workclass")#get_dummies进行ont hot编码
    b = pd.get_dummies(df['education'], prefix="education")
    c = pd.get_dummies(df['marital-status'], prefix="marital-status")
    d = pd.get_dummies(df['occupation'], prefix="occupation")
    e = pd.get_dummies(df['relationship'], prefix="relationship")
    f = pd.get_dummies(df['race'], prefix="race")
    g = pd.get_dummies(df['sex'], prefix="sex")
    h = pd.get_dummies(df['native-country'], prefix="native-country")
    frames = [df, a, b, c, d, e,f,g,h]
    df = pd.concat(frames, axis=1)
    df = df.drop(columns=['workclass','education','marital-status','occupation','relationship','race','sex','native-country'])
    return df


for i in range(len(models)):
    #无处理
    df = pd.read_csv("adult_new.csv")
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化
    df = pd.read_csv("adult_new.csv")
    df=normalization(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化
    df = pd.read_csv("adult_new.csv")
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化
    df = pd.read_csv("adult_new.csv")
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化
    df = pd.read_csv("adult_new.csv")
    df = normalization(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化
    df = pd.read_csv("adult_new.csv")
    df = normalization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化--归一化
    df = pd.read_csv("adult_new.csv")
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化--标准化
    df = pd.read_csv("adult_new.csv")
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化--归一化
    df = pd.read_csv("adult_new.csv")
    df = normalization(df)
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化--标准化
    df = pd.read_csv("adult_new.csv")
    df = normalization(df)
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')


    #无处理  (哑变量)
    df = pd.read_csv("adult_new.csv")
    df = dummyvariable(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')
    #正态化  (哑变量)
    df = pd.read_csv("adult_new.csv")
    df = dummyvariable(df)
    df=normalization(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化  (哑变量)
    df = pd.read_csv("adult_new.csv")
    df = dummyvariable(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化  (哑变量)
    df = pd.read_csv("adult_new.csv")
    df = dummyvariable(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化  (哑变量)
    df = pd.read_csv("adult_new.csv")
    df = dummyvariable(df)
    df = normalization(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化  (哑变量)
    df = pd.read_csv("adult_new.csv")
    df = dummyvariable(df)
    df = normalization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化--归一化  (哑变量)
    df = pd.read_csv("adult_new.csv")
    df = dummyvariable(df)
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化--标准化  (哑变量)
    df = pd.read_csv("adult_new.csv")
    df = dummyvariable(df)
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化--归一化  (哑变量)
    df = pd.read_csv("adult_new.csv")
    df = dummyvariable(df)
    df = normalization(df)
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化--标准化  (哑变量)
    df = pd.read_csv("adult_new.csv")
    df = dummyvariable(df)
    df = normalization(df)
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')
