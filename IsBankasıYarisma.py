# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:47:35 2020

@author: IŞIK
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
sampleSubmission = pd.read_csv("sampleSubmission.csv")

ID = pd.DataFrame(test["ID"])
test_no_ID= test.iloc[:,1:]

x = pd.DataFrame(train["ISLEM_TUTARI"])
y = train.iloc[:,1:]

regressor = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.80, random_state = 0)
regressor.fit(X_train, y_train)
predict = regressor.predict(test_no_ID)