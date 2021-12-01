import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\twoclass-2\crx.csv")#workclass 1836,occupation 1843,native-country 583#32560
for index, row in df.iterrows():
    if(row['A1']=='?' or row['A2']=='?' or row['A4']=='?'or row['A5']=='?'or row['A6']=='?'or row['A7']=='?'or row['A14']=='?'):#删除掉有问号的行
        df.drop(index,inplace = True)
num=0
for index, row in df.iterrows():
    num=num+1;
print(num)
class_mapping = {label:idx for idx,label in enumerate(set(df['A1']))}
df['A1'] = df['A1'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['A4']))}
df['A4'] = df['A4'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['A5']))}
df['A5'] = df['A5'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['A6']))}
df['A6'] = df['A6'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['A7']))}
df['A7'] = df['A7'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['A9']))}
df['A9'] = df['A9'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['A10']))}
df['A10'] = df['A10'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['A12']))}
df['A12'] = df['A12'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['A13']))}
df['A13'] = df['A13'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['A16']))}
df['A16'] = df['A16'].map(class_mapping)
df.to_csv("crx_new.csv")
print(df.head())
