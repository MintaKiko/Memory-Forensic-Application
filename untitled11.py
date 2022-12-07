

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

"""## INPUT DATA"""

plt.style.use('seaborn')

col = ['name','mesin','nomorsection','ptst','nos','sooh','karakter','m','majorlv','minorlv','soc','soi','sou','aoe','boc','ib','sa','fa','majorosv','minorosv','majoriv','minoriv','majorsb','minorsb','win32','sizeoi','sizeoh','ceksum','subsistem','dll','sizeosr','sizeosc','sizeohr','sizeohc','loadf','numberora','label']
df = pd.read_csv('C:/Users/seculab/Desktop/volatility3/volatility3/dataset/dataset/dataset.csv', header=None, names=col)
df.head()

"""## CEK DATA"""

print('Total rows:', df.shape[0])
print('Total columns:', df.shape[1])

df.dtypes

sns.countplot(x='label', data=df)
plt.show()

df.isna()

df.isna().sum()

df.duplicated().sum()

df.drop_duplicates(keep=False,inplace=False)

df.duplicated().sum()

"""## KORELASI"""

plt.subplots(figsize=(40,40))
sns.heatmap(data=df.corr(), annot=True)
plt.show()

df_korelasi = df
df_korelasi = df[['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr','label']]
df_korelasi.head()

boxplot = df_korelasi.boxplot(column=['mesin'])

boxplot = df_korelasi.boxplot(column=['sooh'])

boxplot = df_korelasi.boxplot(column=['m'])

boxplot = df_korelasi.boxplot(column=['ib'])

boxplot = df_korelasi.boxplot(column=['majorsb'])

boxplot = df_korelasi.boxplot(column=['majorosv'])

boxplot = df_korelasi.boxplot(column=['karakter'])

boxplot = df_korelasi.boxplot(column=['sa'])

boxplot = df_korelasi.boxplot(column=['minorosv'])

boxplot = df_korelasi.boxplot(column=['sizeoh'])

boxplot = df_korelasi.boxplot(column=['sizeohr'])

"""## MIN MAX SCALER"""

mms = MinMaxScaler()
df_korelasi = mms.fit_transform(df_korelasi)

col_new = ['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr','label']
df_korelasi = pd.DataFrame(df_korelasi,columns=col_new)

df_korelasi.to_csv('C:/Users/seculab/Desktop/volatility3/volatility3/dataset/dataset/datasetimbangscaling.csv',header=None,index=False)

"""## DECISION TREE SEED"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import tree

usecols=['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr','label']
df_scaling = pd.read_csv('C:/Users/seculab/Desktop/volatility3/volatility3/dataset/dataset/datasetimbangscaling.csv', header=None, names=usecols)
print('Total rows:', df_scaling.shape[0])
print('Total columns:', df_scaling.shape[1])
feature = ['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr']
X = df_scaling[feature] # Features
y = df_scaling.label

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=1) # 70% training and 30% test

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy seed scaling:",metrics.accuracy_score(y_test, y_pred)*100,"%")
print (classification_report(y_test, y_pred))

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('C:/Users/seculab/Desktop/volatility3/hasilscaling.png')
Image(graph.create_png())

"""## DECISION TREE UJI SCALING"""

cols = ['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr']
# df_sampel = pd.read_csv('informasifilezeus.csv', header=None, names=col)
lokasi_uji = input("masukan lokasi file yang uji :")

df_sampel = pd.read_csv(lokasi_uji, header=None, names=col)
sampel = df_sampel[cols]
mms = MinMaxScaler()
sampel = mms.fit_transform(sampel)
col_new = ['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr']
sampel = pd.DataFrame(sampel,columns=col_new)

hasil = clf.predict(sampel)
df_hasil = sampel.assign(Predic = hasil)
print(df_hasil)
print('Keterangan :')
print('Jika Predic Menunjukan 1 adalah Anomali')
print('Jika Predic Menunjukan 0 adalah Non-Anomali')

print(df_hasil['Predic'].unique())
print("Anomali:")
print(df_hasil.loc[df_hasil['Predic'] == 1])

"""## MENGHITUNG AKURASI DECISION TREE UJI SCALING"""

#menghitung akurasi

# df_sampel = pd.read_csv('informasifilezeus.csv', header=None, names=col)
lokasi_uji_akurasi = input("masukan lokasi file yang ingin diiuji akurasi :")

df_hitung = pd.read_csv(lokasi_uji_akurasi, header=None, names=col)
coll = ['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr','label']
hitung = df_hitung[coll]

print("Accuracy dengan scaling:",accuracy_score(hitung.label, hasil.astype(int))*100,"%")
print (classification_report(hitung.label, hasil.astype(int)))

"""## DECISION TREE TIDAK SCALING

"""

col = ['name','mesin','nomorsection','ptst','nos','sooh','karakter','m','majorlv','minorlv','soc','soi','sou','aoe','boc','ib','sa','fa','majorosv','minorosv','majoriv','minoriv','majorsb','minorsb','win32','sizeoi','sizeoh','ceksum','subsistem','dll','sizeosr','sizeosc','sizeohr','sizeohc','loadf','numberora','label']
df = pd.read_csv('C:/Users/seculab/Desktop/volatility3/volatility3/dataset/dataset/dataset.csv', header=None, names=col)
df.head()

df = df[['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr','label']]
feature = ['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr']
X = df[feature] # Features
y = df.label

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=1)

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy seed Tanpa Scaling:",metrics.accuracy_score(y_test, y_pred)*100,"%")
print (classification_report(y_test, y_pred))

cols = ['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr']
# df_sampel = pd.read_csv('informasifilezeus.csv', header=None, names=col)


df_hitung = pd.read_csv(lokasi_uji, header=None, names=col)
# df_sampel = pd.read_csv('dataujikelasnoheader.csv', header=None, names=col)
sampel = df_sampel[cols]

hasil = clf.predict(sampel)
df_hasil = sampel.assign(Predic = hasil)
print(df_hasil)
print('Keterangan :')
print('Jika Predic Menunjukan 1 adalah Anomali')
print('Jika Predic Menunjukan 0 adalah Non-Anomali')
print(df_hasil['Predic'].unique())
print("Anomali:")
print(df_hasil.loc[df_hasil['Predic'] == 1])

"""## MENGHITUNG AKURASI DECISION TREE UJI TIDAK SCALING"""
# lokasi_uji_akurasi = input("masukan lokasi file yang uji ingin diiuji akurasi :")
df_hitung = pd.read_csv(lokasi_uji_akurasi, header=None, names=col)
# df_hitung = pd.read_csv(lokasi_uji_akurasi, header=None, names=col)
# df_hitung = pd.read_csv('dataujikelas.csv', header=None, names=col)
coll = ['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr','label']
hitung = df_hitung[coll]

print("Accuracy dengan tidak scaling:",accuracy_score(hitung.label, hasil)*100,"%")
print (classification_report(hitung.label, hasil))
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('C:/Users/seculab/Desktop/volatility3/hasil.png')
Image(graph.create_png())