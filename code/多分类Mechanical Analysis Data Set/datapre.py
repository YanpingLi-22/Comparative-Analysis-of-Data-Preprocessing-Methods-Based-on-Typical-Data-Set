import numpy as np
import pandas as pd

df = pd.read_csv("mechanical-analysis.csv")#workclass 1836,occupation 1843,native-country 583#32560

class_mapping = {label:idx for idx,label in enumerate(set(df['dir']))}
df['dir'] = df['dir'].map(class_mapping)


df.to_csv("mechanical-analysis_new.csv")
print(df.head())
