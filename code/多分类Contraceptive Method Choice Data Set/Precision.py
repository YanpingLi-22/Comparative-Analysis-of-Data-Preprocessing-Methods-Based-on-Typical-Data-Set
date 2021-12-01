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

# models = [neutralnetwork_model]
# models_Name = ['logistic_model','decisiontree_model','randomforest_model','svm_model','gaussion_model','KNN_model']
#正态化
def normalization(df):
    #print(df["Height"])
    df["WAge"] = yeojohnson(df["WAge"])[0]
    df["NumOfBorn"] = yeojohnson(df["NumOfBorn"])[0]
    return df
#标准化
def standardization(df):
    from sklearn.preprocessing import StandardScaler
    StandardScaler = StandardScaler()
    columns_to_scale = ['WAge','NumOfBorn']
    df[columns_to_scale] = StandardScaler.fit_transform(df[columns_to_scale])
    return df
#归一化
def preprocessing(df):
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    columns_to_scale = ['WAge','NumOfBorn']
    df[columns_to_scale] = min_max_scaler.fit_transform(df[columns_to_scale])
    return df
#哑变量
def dummyvariable(df):
    a = pd.get_dummies(df['WEducation'], prefix = "WEducation")#get_dummies进行ont hot编码
    b = pd.get_dummies(df['HEducation'], prefix="HEducation")
    c = pd.get_dummies(df['WReligion'], prefix="WReligion")
    d = pd.get_dummies(df['WiWork'], prefix="WiWork")
    e = pd.get_dummies(df['HusOcc'], prefix="HusOcc")
    f = pd.get_dummies(df['SOLI'], prefix="SOLI")
    g = pd.get_dummies(df['MEB'], prefix="MEB")
    frames = [df, a, b, c, d, e, f, g]
    df = pd.concat(frames, axis=1)
    df = df.drop(columns=['WEducation','HEducation','WReligion','WiWork','HusOcc','SOLI','MEB'])
    return df

for i in range(len(models)):
    #无处理
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    print(111111111111)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')
    #正态化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df=normalization(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = normalization(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = normalization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化--归一化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化--标准化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化--归一化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = normalization(df)
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化--标准化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = normalization(df)
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')


    #无处理  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = dummyvariable(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')
    #正态化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = dummyvariable(df)
    df=normalization(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = dummyvariable(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = dummyvariable(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = dummyvariable(df)
    df = normalization(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = dummyvariable(df)
    df = normalization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化--归一化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = dummyvariable(df)
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化--标准化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = dummyvariable(df)
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化--归一化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = dummyvariable(df)
    df = normalization(df)
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化--标准化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
    df = dummyvariable(df)
    df = normalization(df)
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['CMUC'], axis=1)
    y = df['CMUC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result,y_score=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result, y_score,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

