#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 18:09:16 2022

@author: cumalitoprak
"""

#kutuphaneler
import pandas as pd #veriler yuklemek icin ,verileri düzgün bir şekilde tutabilmek için kullanılır
import numpy as np #büyük hesaplamalar ve sayılar için kullanılır
import matplotlib.pyplot as plt #çizim için kullanılır

#kodlar
#veri yukleme
veriler = pd.read_csv("eksikveriler.csv") #veriler.csv aynı dizin içerisinde ise, absolute path
print(veriler)

#veri ön işleme
boy = veriler[["boy"]]
print(boy)

boykilo = veriler[["boy", "kilo"]]
print(boykilo)

#eksik veriler
#sklearn -> kutuphane, impute -> modul, SimpleImputer -> class
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #mean ortalama deger, yani nan(not a number) degerleri, ortalama deger ile degistir
Yas = veriler.iloc[:,1:4].values #: -> tüm satırlar, 1:4 -> 1 den 4'e kadar olan kolonlar #iloc: integer location fonksiyonu
print(Yas)
imputer = imputer.fit(Yas[:,1:4]) #fit ile ogreniyoruz, neyi nan degerlerinin yerine gelecek ortalama degeleri ogreniyoruz
Yas[:,1:4] = imputer.transform(Yas[:,1:4]) #transform ile eksik verileri dönüştürüyoruz
print(Yas)


ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder(); #label enconding işlemi

ulke[:, 0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

print(list(range(22)))
sonuc = pd.DataFrame(data=ulke, index = range(22), columns=['fr', 'tr', 'us']) #dataframe olusturma
print(sonuc)
sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns=['boy', 'kilo', 'yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:, -1].values # -1 sondan 1 önceki kolon demek
print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])
print(cinsiyet)
                                  
#olusturdugumuz dataframeleri birlestirme

s = pd.concat([sonuc, sonuc2], axis=1);
print(s)
s2 = pd.concat([s, sonuc3], axis=1)
print(s2)







#model olusturmak