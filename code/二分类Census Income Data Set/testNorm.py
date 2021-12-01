import pandas as pd
df = pd.DataFrame([
   ['green', 'M', 10.1, 'class1'],
   ['red', 'L', 13.5, 'class2'],
   ['blue', 'XL', 15.3, 'class1'],
])

df.columns = ['color', 'size', 'prize', 'class label']
#
# size_mapping = {
#    'XL': 3,
#    'L': 2,
#    'M': 1}
# df['size'] = df['size'].map(size_mapping)

class_mapping = {label:idx for idx,label in enumerate(set(df['size']))}
df['size'] = df['size'].map(class_mapping)
print(df.head())

# df['workclass'] = df['workclass'].replace('Private',0)
# df['workclass'] = df['workclass'].replace('Self-emp-not-inc',1)
# df['workclass'] = df['workclass'].replace('Self-emp-inc',2)
# df['workclass'] = df['workclass'].replace('Federal-gov',3)
# df['workclass'] = df['workclass'].replace('Local-gov',4)
# df['workclass'] = df['workclass'].replace('State-gov',5)
# df['workclass'] = df['workclass'].replace('Without-pay',6)
# df['workclass'] = df['workclass'].replace('Never-worked',7)
#
# df['education'] = df['education'].replace('Bachelors',0)
# df['education'] = df['education'].replace('Some-college',1)
# df['education'] = df['education'].replace('11th',2)
# df['education'] = df['education'].replace('HS-grad',3)
# df['education'] = df['education'].replace('Prof-school',4)
# df['education'] = df['education'].replace('Assoc-acdm',5)
# df['education'] = df['education'].replace('9th',6)
# df['education'] = df['education'].replace('7th-8th',7)
# df['education'] = df['education'].replace('12th',8)
# df['education'] = df['education'].replace('Masters',9)
# df['education'] = df['education'].replace('1st-4th',10)
# df['education'] = df['education'].replace('10th',11)
# df['education'] = df['education'].replace('Doctorate',12)
# df['education'] = df['education'].replace('5th-6th',13)
# df['education'] = df['education'].replace('Preschool',14)
#
# df['marital-status'] = df['marital-status'].replace('Married-civ-spouse',0)
# df['marital-status'] = df['marital-status'].replace('Divorced',1)
# df['marital-status'] = df['marital-status'].replace('Never-married',2)
# df['marital-status'] = df['marital-status'].replace('Separated',3)
# df['marital-status'] = df['marital-status'].replace('Widowed',4)
# df['marital-status'] = df['marital-status'].replace('Married-spouse-absent',5)
# df['marital-status'] = df['marital-status'].replace('Married-AF-spouse',6)
#
# df['occupation'] = df['occupation'].replace('Tech-support',0)
# df['occupation'] = df['occupation'].replace('Craft-repair',1)
# df['occupation'] = df['occupation'].replace('Other-service',2)
# df['occupation'] = df['occupation'].replace('Sales',3)
# df['occupation'] = df['occupation'].replace('Exec-managerial',4)
# df['occupation'] = df['occupation'].replace('Prof-specialty',5)
# df['occupation'] = df['occupation'].replace('Handlers-cleaners',6)
# df['occupation'] = df['occupation'].replace('Machine-op-inspct',7)
# df['occupation'] = df['occupation'].replace('Adm-clerical',8)
# df['occupation'] = df['occupation'].replace('Farming-fishing',9)
# df['occupation'] = df['occupation'].replace('Transport-moving',10)
# df['occupation'] = df['occupation'].replace('Priv-house-serv',11)
# df['occupation'] = df['occupation'].replace('Protective-serv',12)
# df['occupation'] = df['occupation'].replace('Armed-Forces',14)
#
# df['relationship'] = df['relationship'].replace('Wife',0)
# df['relationship'] = df['relationship'].replace('Own-child',1)
# df['relationship'] = df['relationship'].replace('Not-in-family',2)
# df['relationship'] = df['relationship'].replace('Other-relative',3)
# df['relationship'] = df['relationship'].replace('Unmarried',4)
# df['relationship'] = df['relationship'].replace('Husband',5)
#
# df['race'] = df['race'].replace('White',0)
# df['race'] = df['race'].replace('Asian-Pac-Islander',1)
# df['race'] = df['race'].replace('Amer-Indian-Eskimo',2)
# df['race'] = df['race'].replace('Black',3)
# df['race'] = df['race'].replace('Other',4)
#
# df['sex'] = df['sex'].replace('Female',1)
# df['sex'] = df['sex'].replace('Female',0)
#
# df['native-country'] = df['native-country'].replace('Black',0)
# df['native-country'] = df['native-country'].replace('Other',1)
# df['native-country'] = df['native-country'].replace('Black',2)
# df['native-country'] = df['native-country'].replace('Other',3)
# df['native-country'] = df['native-country'].replace('Black',4)
# df['native-country'] = df['native-country'].replace('Other',5)
# df['native-country'] = df['native-country'].replace('Black',6)
# df['native-country'] = df['native-country'].replace('Other',7)
# df['native-country'] = df['native-country'].replace('Black',8)
# df['native-country'] = df['native-country'].replace('Other',9)
# df['native-country'] = df['native-country'].replace('Black',10)
# df['native-country'] = df['native-country'].replace('Other',11)
# df['native-country'] = df['native-country'].replace('Black',12)
# df['native-country'] = df['native-country'].replace('Other',13)