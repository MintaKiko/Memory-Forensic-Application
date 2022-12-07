import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score



col = ['name','mesin','nomorsection','ptst','nos','sooh','karakter','m','majorlv','minorlv','soc','soi','sou','aoe','boc','ib','sa','fa','majorosv','minorosv','majoriv','minoriv','majorsb','minorsb','win32','sizeoi','sizeoh','ceksum','subsistem','dll','sizeosr','sizeosc','sizeohr','sizeohc','loadf','numberora','label']
df = pd.read_csv('C:/Users/seculab/Desktop/volatility3/volatility3/dataset/dataset/dataset.csv', header=None, names=col)
df.head()

df = df[['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr','label']]
feature = ['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr']
# X = df[feature] # Features

X_train = df[feature]
y_train = df.label
# y_test = df.label
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=1)
X_test = pd.read_csv('C:/Users/seculab/Desktop/volatility3/volatility3/dataset/dataset/dataxtest.csv',header=None, names= feature)
# y_train = pd.read_csv('C:/Users/seculab/Desktop/volatility3/volatility3/dataset/dataset/dataytrain.csv',header=None)
y_test = pd.read_csv('C:/Users/seculab/Desktop/volatility3/volatility3/dataset/dataset/dataytrain.csv',header=None)

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

# print("Accuracy seed Tanpa Scaling:",metrics.accuracy_score(df.label, y_pred)*100,"%")
# print (classification_report(df.label, y_pred))

cols = ['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr']
# df_sampel = pd.read_csv('informasifilezeus.csv', header=None, names=col)

lokasi_uji_akurasi = input("masukan lokasi file yang uji ingin diiuji akurasi :")
df_hitung = pd.read_csv(lokasi_uji_akurasi, header=None, names=col)
# df_hitung = pd.read_csv(lokasi_uji, header=None, names=col)
# df_sampel = pd.read_csv('dataujikelasnoheader.csv', header=None, names=col)
sampel = df_hitung[cols]

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
lokasi_uji_akurasi = input("masukan lokasi file yang uji ingin diiuji akurasi :")
df_hitung = pd.read_csv(lokasi_uji_akurasi, header=None, names=col)
# df_hitung = pd.read_csv(lokasi_uji_akurasi, header=None, names=col)
# df_hitung = pd.read_csv('dataujikelas.csv', header=None, names=col)
coll = ['mesin','sooh','m','ib','majorsb','majorosv','karakter','sa','minorosv','sizeoh','sizeohr','label']
hitung = df_hitung[coll]

print("Accuracy dengan tidak scaling:",accuracy_score(hitung.label, hasil)*100,"%")
print (classification_report(hitung.label, hasil))