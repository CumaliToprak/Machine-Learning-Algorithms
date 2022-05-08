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
veriler = pd.read_csv("veriler.csv") #veriler.csv aynı dizin içerisinde ise, absolute path
#print(veriler)
boy = veriler[["boy"]]
print(boy)

boykilo = veriler[["boy", "kilo"]]
print(boykilo)
#veri ön işleme

#model olusturmak