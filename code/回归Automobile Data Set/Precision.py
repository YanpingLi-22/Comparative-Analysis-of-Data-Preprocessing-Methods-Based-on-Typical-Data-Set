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

# models = [SVM_model,Linear_model,DTree_model,KNN_model,RandomForest_model,AdaBoost_model,GBRT_model,Bagging_model,ExtraTree_model,ARD_model]
# models_Name = ['SVM_model','Linear_model','DTree_model','KNN_model','RandomForest_model','AdaBoost_model','GBRT_model','Bagging_model','ExtraTree_model','ARD_model']

models = [SVM_model]
models_Name = ['SVM_model']
#正态化
def normalization(df):
    df["length"] = yeojohnson(df["length"])[0]
    df["width"] = yeojohnson(df["width"])[0]
    df["height"] = yeojohnson(df["height"])[0]
    df["curb-weight"] = yeojohnson(df["curb-weight"])[0]
    df["wheel-base"] = yeojohnson(df["wheel-base"])[0]
    df["engine-size"] = yeojohnson(df["engine-size"])[0]
    df["stroke"] = yeojohnson(df["stroke"])[0]
    df["bore"] = yeojohnson(df["bore"])[0]
    df["compression-ratio"] = yeojohnson(df["compression-ratio"])[0]
    df["horsepower"] = yeojohnson(df["horsepower"])[0]
    df["highway-mpg"] = yeojohnson(df["highway-mpg"])[0]
    df["city-mpg"] = yeojohnson(df["city-mpg"])[0]
    df["peak-rpm"] = yeojohnson(df["peak-rpm"])[0]
    return df
#标准化
def standardization(df):
    from sklearn.preprocessing import StandardScaler
    StandardScaler = StandardScaler()
    columns_to_scale = ['length','width','height','curb-weight','wheel-base','engine-size','stroke','bore','compression-ratio','horsepower','highway-mpg','city-mpg','peak-rpm']
    df[columns_to_scale] = StandardScaler.fit_transform(df[columns_to_scale])
    return df
#归一化
def preprocessing(df):
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    columns_to_scale = ['length','width','height','curb-weight','wheel-base','engine-size','stroke','bore','compression-ratio','horsepower','highway-mpg','city-mpg','peak-rpm']
    df[columns_to_scale] = min_max_scaler.fit_transform(df[columns_to_scale])
    return df
#哑变量
def dummyvariable(df):
    a = pd.get_dummies(df['symboling'], prefix = "symboling")#get_dummies进行ont hot编码
    b = pd.get_dummies(df['make'], prefix="make")
    c = pd.get_dummies(df['fuel-type'], prefix="fuel-type")
    d = pd.get_dummies(df['aspiration'], prefix="aspiration")
    e = pd.get_dummies(df['num-of-doors'], prefix="num-of-doors")
    f = pd.get_dummies(df['body-style'], prefix="body-style")
    g = pd.get_dummies(df['drive-wheels'], prefix="drive-wheels")
    h = pd.get_dummies(df['engine-location'], prefix="engine-location")
    i = pd.get_dummies(df['engine-type'], prefix="engine-type")
    j = pd.get_dummies(df['num-of-cylinders'], prefix="num-of-cylinders")
    k = pd.get_dummies(df['fuel-system'], prefix="fuel-system")
    frames = [df, a, b, c, d, e, f, g,h,i,j,k]
    df = pd.concat(frames, axis=1)
    df = df.drop(columns=['symboling','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','engine-type','num-of-cylinders','fuel-system'])
    return df

for i in range(len(models)):

    #无处理
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"]/1000
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')


    #正态化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"]/1000
    df=normalization(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"]/1000
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = normalization(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = normalization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化--归一化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化--标准化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化--归一化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = normalization(df)
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化--标准化
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = normalization(df)
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')


    #无处理  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = dummyvariable(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')
    #正态化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = dummyvariable(df)
    df=normalization(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = dummyvariable(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = dummyvariable(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = dummyvariable(df)
    df = normalization(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = dummyvariable(df)
    df = normalization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #标准化--归一化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = dummyvariable(df)
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #归一化--标准化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = dummyvariable(df)
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--标准化--归一化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = dummyvariable(df)
    df = normalization(df)
    df = standardization(df)
    df = preprocessing(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

    #正态化--归一化--标准化  (哑变量)
    df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
    df["price"] = df["price"] / 1000
    df = dummyvariable(df)
    df = normalization(df)
    df = preprocessing(df)
    df = standardization(df)
    base_model = models[i]
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    result=base_model(X_train, X_test,y_train, y_test)
    eva = evaluate(result,models_Name[i])
    Re = eva.show_all()
    Re.to_csv('1.txt', mode='a', float_format='%.4f')

