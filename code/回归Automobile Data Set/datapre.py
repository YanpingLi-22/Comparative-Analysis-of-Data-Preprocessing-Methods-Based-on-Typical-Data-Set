import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\imports-85.csv")#workclass 1836,occupation 1843,native-country 583#32560
for index, row in df.iterrows():
    if(row['num-of-doors']=='?' or row['bore']=='?' or row['stroke']=='?' or row['horsepower']=='?' or row['peak-rpm']=='?' or row['price']=='?'):#删除掉有问号的行
        df.drop(index,inplace = True)
# num=0
# for index, row in df.iterrows():
#     num=num+1
# print(num)
class_mapping = {label:idx for idx,label in enumerate(set(df['make']))}
df['make'] = df['make'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['fuel-type']))}
df['fuel-type'] = df['fuel-type'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['aspiration']))}
df['aspiration'] = df['aspiration'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['num-of-doors']))}
df['num-of-doors'] = df['num-of-doors'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['body-style']))}
df['body-style'] = df['body-style'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['drive-wheels']))}
df['drive-wheels'] = df['drive-wheels'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['engine-location']))}
df['engine-location'] = df['engine-location'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['engine-type']))}
df['engine-type'] = df['engine-type'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['num-of-cylinders']))}
df['num-of-cylinders'] = df['num-of-cylinders'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['fuel-system']))}
df['fuel-system'] = df['fuel-system'].map(class_mapping)
#
df.to_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")
# print(df.head())
