import os 
import pandas as pd
import numpy as np
import csv 



path = "data"

trainers = pd.read_csv("my-collection.csv")
trainers.columns
trainers.head(100)

#trainers["Colour"].value_counts() 
trainers["Release Date"].value_counts() 
print(trainers["Secondary Market Price (UK8 Stock X)"])

trainers["Cost"] = trainers["Cost"].str.replace("£","").astype(float)
trainers.head(100)

trainers["Retail Price"] = trainers["Retail Price"].str.replace("£","").astype(float)
trainers.head(100)

trainers["Designer"].value_counts() 
print(trainers["Colour"])
trainers.head(5)

trainers["Secondary Market Price (UK8 Stock X)"] = trainers["Secondary Market Price (UK8 Stock X)"].str.replace(",","").str.replace("£","").astype(float)
trainers.head(10)

trainers["Price difference"] = trainers["Secondary Market Price (UK8 Stock X)"] - trainers ["Cost"]
trainers.describe()

trainers["Price difference"] > 0

sum(trainers["Price difference"] > 0)

import matplotlib.pyplot as plt

n, bins, patches = plt.hist(trainers["Price difference"])

plt.xlabel('Difference in Price')
plt.ylabel('Frequency')
plt.title('Histogram of Price Differences')

trainers.head()

x=trainers["Cost"]
y=trainers["Secondary Market Price (UK8 Stock X)"]



plt.scatter(x,y)

plt.xlabel('Cost')
plt.ylabel('Seconday market price')
plt.title('Scatter graph of price change')

plt.xlim(0,1800)
plt.ylim(0,1850)
plt.axline((0,0),(1,1))

trainers["Years owned"] = 2021 - trainers["Purchase Year"] + 1
trainers["Price rate"]=trainers["Price difference"] / trainers["Years owned"]

n, bins, patches = plt.hist(trainers["Price rate"])

plt.xlabel('Raate of difference in price')
plt.ylabel('Frequency')
plt.title('Histogram of rate of price Differences')

trainers.head(5)

import matplotlib
import matplotlib.pyplot as plt
import pandas as panda
import numpy as np

from sklearn import linear_model
#Create linear regression object
regr = linear_model.LinearRegression()
x=trainers["Cost"].values.reshape(-1, 1)
y=trainers["Secondary Market Price (UK8 Stock X)"]
regr.fit(x,y)


plt.scatter(x,y)

plt.xlabel('Cost')
plt.ylabel('Seconday market price')
plt.title('Scatter graph of price change')
x_min = 0 
x_max = 400
y_pred = regr.predict(np.array([x_min, x_max]).reshape(-1,1))
y_min = y_pred[0]
y_max = y_pred[1]
plt.axline((x_min,y_min),(x_max,y_max))

print("Coefficients: \n", regr.coef_)
print("intercept: \n", regr.intercept_)

trainers.head()

assert (trainers["Shoe Size (UK 8)"] < 9).all(), "Atleast one pair of shoes is size 9 or above"
assert (trainers["Shoe Size (UK 8)"] > 7).all(), "Atleast one pair of shoes is size 7 or below"