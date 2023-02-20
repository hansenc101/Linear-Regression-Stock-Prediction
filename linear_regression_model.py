# -*- coding: utf-8 -*-
"""
Christopher Hansen
Artificial Neural Networks

"""
#%% Prepare The Data
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_price(tick,start='2021-10-01',end=None):
    return yf.Ticker(tick).history(start=start,end=end)['Close']

def get_prices(tickers,start='2021-10-01',end='2022-04-02'):
    df=pd.DataFrame()
    for s in tickers:
        df[s]=get_price(s,start,end)
    return df

# Prepare Training and Data Sets
feature_stocks=['tsla','meta','amzn','nflx','gbtc','gdx','intc','dal','c']
predict_stock='msft'

# training set
start_date_train='2021-10-01'
end_date_train='2021-12-31'

X_train=get_prices(feature_stocks,start=start_date_train,end=end_date_train)
y_train=get_prices([predict_stock],start=start_date_train,end=end_date_train)

# testing set
start_date_test='2022-01-01' # end date omit, default is doday
X_test=get_prices(feature_stocks,start=start_date_test)
y_test=get_prices([predict_stock],start=start_date_test)

# Convert Training and Testing data into nympy array
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)

#%% 1. Append a dummy feature to both X_train and X_test
n_dummy_features = X_train.shape[0] 
dummy_features = np.ones((n_dummy_features,1))
X_train = np.append(X_train, dummy_features, 1)
X_test = np.append(X_test, dummy_features, 1)

#%% 2. Find the best linear regression model based on your training data 
# w = X*y*(XX′)^(−1)
result1 = np.linalg.pinv(np.dot(X_train.T,X_train))
result2 = np.dot(X_train,result1)
weights = np.dot(result2.T,y_train)

#%% 3. Report your training and testing error
# How far your prediction from the actual price. Compute the mean square error 
# for both training and testing
def compute_mse(w,x_test,y_test):
    y=np.dot(x_test,w)
    mse=np.dot((y-y_test).transpose(),(y-y_test))/y.size
    if len(mse.shape)==2:
        mse=mse[0][0]
    
    return mse,y

training_mse = compute_mse(weights,X_train,y_train)
testing_mse = compute_mse(weights,X_test,y_test)
print(f"The MSE for training data is: {training_mse[0]}")
print(f"The MSE for testing data is: {testing_mse[0]}")

# I am plotting to see what everything looks like
x = np.linspace(1,63,63)
y=np.dot(X_test,weights)
plt.plot(x, y_test, x, y)
plt.title("Test Linear Regression against Test Data")


y2 = np.dot(X_train,weights)
plt.figure()
plt.plot(x, y_train, x, y2)
plt.title("Test Linear Regression agains Training Data")
plt.show()