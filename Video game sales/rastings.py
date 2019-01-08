import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

games = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')
print(games.head())
print(games.Platform.value_counts())
print(games.Name[np.isnan(games.Critic_Score)])
print(np.isnan(games.Critic_Score).sum())
games_reduced = games[games.Critic_Score.notnull()]
print(games_reduced)
print(games_reduced.Platform.value_counts())
ps2_reduced=games_reduced[games_reduced.Platform=='PS2']
gc_reduced=games_reduced[games_reduced.Platform=='GC']
xb_reduced=games_reduced[games_reduced.Platform=='XB']
ps2x=np.asarray(preprocessing.scale(ps2_reduced.Critic_Score))
ps2y=ps2_reduced.Global_Sales
ps2x_train,ps2x_test,ps2y_train,ps2y_test = train_test_split(ps2x,ps2y,test_size=0.2)
ps2x_train = ps2x_train.reshape(-1,1)
ps2y_train = ps2y_train.values.reshape(-1,1)
ps2x_test = ps2x_test.reshape(-1,1)
ps2y_test = ps2y_test.values.reshape(-1,1)
ps2regr = linear_model.LinearRegression()
ps2regr.fit(ps2x_train,ps2y_train)
ps2y_pred = ps2regr.predict(ps2x_test)
print('Coefficients: ', ps2regr.coef_)
print('Mean square error: ', mean_squared_error(ps2y_test, ps2y_pred))
print('Variance: ', r2_score(ps2y_test,ps2y_pred))
plt.scatter(ps2x_test,ps2y_test,c='black')
plt.plot(ps2x_test,ps2y_pred,c='black')
gcx=np.asarray(preprocessing.scale(gc_reduced.Critic_Score))
gcy=gc_reduced.Global_Sales
gcx_train,gcx_test,gcy_train,gcy_test = train_test_split(gcx,gcy,test_size=0.2)
gcx_train = gcx_train.reshape(-1,1)
gcy_train = gcy_train.values.reshape(-1,1)
gcx_test = gcx_test.reshape(-1,1)
gcy_test = gcy_test.values.reshape(-1,1)
gcregr = linear_model.LinearRegression()
gcregr.fit(gcx_train,gcy_train)
gcy_pred = gcregr.predict(gcx_test)
print('Coefficients: ', gcregr.coef_)
print('Mean square error: ', mean_squared_error(gcy_test, gcy_pred))
print('Variance: ', r2_score(gcy_test,gcy_pred))
plt.scatter(gcx_test,gcy_test,c='blue')
plt.plot(gcx_test,gcy_pred,c='blue')
xbx=np.asarray(preprocessing.scale(xb_reduced.Critic_Score))
xby=xb_reduced.Global_Sales
xbx_train,xbx_test,xby_train,xby_test = train_test_split(xbx,xby,test_size=0.2)
xbx_train = xbx_train.reshape(-1,1)
xby_train = xby_train.values.reshape(-1,1)
xbx_test = xbx_test.reshape(-1,1)
xby_test = xby_test.values.reshape(-1,1)
xbregr = linear_model.LinearRegression()
xbregr.fit(xbx_train,xby_train)
xby_pred = xbregr.predict(xbx_test)
print('Coefficients: ', xbregr.coef_)
print('Mean square error: ', mean_squared_error(xby_test, xby_pred))
print('Variance: ', r2_score(xby_test,xby_pred))
plt.scatter(xbx_test,xby_test,c='green')
plt.plot(xbx_test,xby_pred,c='green')
plt.show()