import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

col = ['name','mesin','nomorsection','ptst','nos','sooh','karakter','m','majorlv','minorlv','soc','soi','sou','aoe','boc','ib','sa','fa','majorosv','minorosv','majoriv','minoriv','majorsb','minorsb','win32','sizeoi','sizeoh','ceksum','subsistem','dll','sizeosr','sizeosc','sizeohr','sizeohc','loadf','numberora','label']
df = pd.read_csv('C:/Users/Digital Forensic/Desktop/volatility3/dataset/dataset/dataset.csv', header=None, names=col)

# df = df[['nomorsection','ptst','nos','karakter','soc','soi','sou','aoe','boc','sa','fa','minorosv','majoriv','minoriv','minorsb','win32','sizeoi','sizeoh','ceksum','subsistem','dll','sizeosr','sizeosc','sizeohr','sizeohc','numberora','label']]
# df = df[['sizeoh','sizeosr','minorosv','fa','label']]
df = df[['karakter','minorosv','majoriv','minoriv','minorsb','sizeosc','sizeohr','sizeohc','label']]

# check_for_nan = df.isnull()
# print (check_for_nan)
# print (df)

#split dataset in features and target variable
# feature = ['nomorsection','ptst','nos','karakter','soc','soi','sou','aoe','boc','sa','fa','minorosv','majoriv','minoriv','minorsb','win32','sizeoi','sizeoh','ceksum','subsistem','dll','sizeosr','sizeosc','sizeohr','sizeohc','numberora']
# feature = ['sizeoh','sizeosr','minorosv','fa']
feature = ['karakter','minorosv','majoriv','minoriv','minorsb','sizeosc','sizeohr','sizeohc']
X = df[feature] # Features
y = df.label # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0) # 70% training and 30% test
# print(X_train)
# print("ini ",X_test)
# print(y_train)
# y_test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

#menghitung akurasi
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100,"%")
print (classification_report(y_test, y_pred))

# cols = ['nomorsection','ptst','nos','karakter','soc','soi','sou','aoe','boc','sa','fa','minorosv','majoriv','minoriv','minorsb','win32','sizeoi','sizeoh','ceksum','subsistem','dll','sizeosr','sizeosc','sizeohr','sizeohc','numberora']
# cols = ['sizeoh','sizeosr','minorosv','fa']
cols = ['karakter','minorosv','majoriv','minoriv','minorsb','sizeosc','sizeohr','sizeohc']
# df_sampel = pd.read_csv('informasifilezeus.csv', header=None, names=col)
lokasi = input("masukan lokasi file yang ingin diiuji :")
# df_sampel = pd.read_csv('C:/Users/Digital Forensic/Desktop/volatility3/dataset/data pengujian/informasifilecridex.csv', header=None, names=col)
df_sampel = pd.read_csv(lokasi, header=None, names=col)
sampel = df_sampel[cols]

hasil = clf.predict(sampel)
df_hasil = sampel.assign(Predic = hasil)
print(df_hasil)
print("Keterangan :")
print("Jika Predic menunjukan 1 adalah anomali")
print('jika Predic menunjukan 0 adalah non anomali')
print(df_hasil['Predic'].unique())
print("Anomali:")
print(df_hasil.loc[df_hasil['Predic'] == 1])