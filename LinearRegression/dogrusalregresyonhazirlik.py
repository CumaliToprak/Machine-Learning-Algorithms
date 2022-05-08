# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('satislar.csv')
#test
print(veriler)

# veri on isleme
aylar = veriler[["Aylar"]] #bağımsız değişkenler - [:,:1] is eq to [tüm satırlar: 0:1], 0:1 ilk kolon
print(aylar)
satislar = veriler[["Satislar"]] #bağımlı değişken
print(satislar)


#yukarı kolonlarına ayırma kısmını asagıdaki gibi de yapabiliriz
'''
aylar = veriler.iloc[:,:1].values #bağımsız değişkenler - [:,:1] is eq to [tüm satırlar: 0:1], 0:1 ilk kolon
print(aylar)
satislar = veriler.iloc[:,1:2].values #bağımlı değişken
print(satislar)
'''


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)
'''
#verilerin olceklenmesi, veriler standardize ediliyor
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.transform(y_test)
'''
#Modeli inşa ettiğimiz yer(linear regression)
from sklearn.linear_model import LinearRegression 

lr = LinearRegression() #linear regression objesi oluşturma
lr.fit(x_train, y_train)


#modelin uygulanması - tahmin yapma
tahmin = lr.predict(x_test)

#sonucların görsellestirilmesi

#öncesinde verileri sortlamamız lazım yoksa grafik bozuk olur, aylar sıralı degil..
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))

plt.title("Aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")

'''
# multiline comment

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)


'''














