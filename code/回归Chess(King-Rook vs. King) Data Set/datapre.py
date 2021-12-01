import numpy as np
import pandas as pd

df = pd.read_csv("krkopt.csv")#workclass 1836,occupation 1843,native-country 583#32560

df['WIN'] = df['WIN'].replace('draw',-1)
df['WIN'] = df['WIN'].replace('zero',0)
df['WIN'] = df['WIN'].replace('one',1)
df['WIN'] = df['WIN'].replace('two',2)
df['WIN'] = df['WIN'].replace('three',3)
df['WIN'] = df['WIN'].replace('four',4)
df['WIN'] = df['WIN'].replace('five',5)
df['WIN'] = df['WIN'].replace('six',6)
df['WIN'] = df['WIN'].replace('seven',7)
df['WIN'] = df['WIN'].replace('eight',8)
df['WIN'] = df['WIN'].replace('nine',9)
df['WIN'] = df['WIN'].replace('ten',10)
df['WIN'] = df['WIN'].replace('eleven',11)
df['WIN'] = df['WIN'].replace('twelve',12)
df['WIN'] = df['WIN'].replace('thirteen',13)
df['WIN'] = df['WIN'].replace('fourteen',14)
df['WIN'] = df['WIN'].replace('fifteen',15)
df['WIN'] = df['WIN'].replace('sixteen',16)

class_mapping = {label:idx for idx,label in enumerate(set(df['WKf']))}
df['WKf'] = df['WKf'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['WRf']))}
df['WRf'] = df['WRf'].map(class_mapping)
class_mapping = {label:idx for idx,label in enumerate(set(df['BKf']))}
df['BKf'] = df['BKf'].map(class_mapping)

df.to_csv("krkopt-new.csv")
# print(df.head())
