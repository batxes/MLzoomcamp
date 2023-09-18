# 17 September

import pandas as pd
import numpy as np

#question 1
print (pd.__version__)

#question 2
df = pd.read_csv("housing.csv",sep=",")
print (df)

#question 3
print (df.describe())
print (df.isnull().sum())

#question 4
print (df.ocean_proximity.unique())
print (len(df.ocean_proximity.unique()))

#question 5
print (df.query('ocean_proximity == "NEAR BAY"').median_house_value.mean().round())

#question 6
avg = df.total_bedrooms.mean()
print (avg)
df["total_bedrooms"] = df["total_bedrooms"].fillna(avg)
print (df.isnull().sum())
avg = df.total_bedrooms.mean()
print (avg)

#question 7
print ("question 7")
islands = df.query('ocean_proximity == "ISLAND"')
islands = islands[["housing_median_age", "total_rooms", "total_bedrooms"]]
X = islands.to_numpy()
print (X)

# the same as u.dot(v)
def vector_vector_multiplication(u,v):
    assert u.shape[0] == v.shape[0]
    n = u.shape[0]
    result = 0.0
    for i in range(n):
        result = result + u[i] * v[i]
    return result

#the same as U.dot(v)
def matrix_vector_multiplication(U,v):
    assert U.shape[1] == v.shape[0]
    num_rows =U.shape[0]
    result = np.zeros(num_rows)
    for i in range (num_rows):
        result[i] = vector_vector_multiplication(U[i],v)
    return result

#the same as U.dot(V)
def matrix_matrix_multiplication(U,V):
    assert U.shape[1] == V.shape[0]
    num_rows =U.shape[0]
    num_cols =V.shape[1]
    result = np.zeros((num_rows,num_cols))
    for i in range (num_cols):
        vi = V[:,i]
        Uvi = matrix_vector_multiplication(U,vi)
        result[:,i] = Uvi 
    return result

XTX = (X.T).dot(X)
XTX = matrix_matrix_multiplication(X.T,X)

print (XTX == XTX)

y = np.array([950, 1300, 800, 1000, 1300])

w = np.linalg.inv(XTX).dot(X.T).dot(y)
print (w)

