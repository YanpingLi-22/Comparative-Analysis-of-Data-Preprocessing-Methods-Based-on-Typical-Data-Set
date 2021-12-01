import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\twoclass\adult.csv")#workclass 1836,occupation 1843,native-country 583#32560
for index, row in df.iterrows():
    if(row['native-country']=='?' or row['occupation']=='?' or row['workclass']=='?'):#删除掉有问号的行
        df.drop(index,inplace = True)

class_mapping = {label:idx for idx,label in enumerate(set(df['workclass']))}
df['workclass'] = df['workclass'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['education']))}
df['education'] = df['education'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['marital-status']))}
df['marital-status'] = df['marital-status'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['occupation']))}
df['occupation'] = df['occupation'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['relationship']))}
df['relationship'] = df['relationship'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['race']))}
df['race'] = df['race'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['sex']))}
df['sex'] = df['sex'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['native-country']))}
df['native-country'] = df['native-country'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['income']))}
df['income'] = df['income'].map(class_mapping)

df.to_csv("adult_new.csv")
print(df.head())
