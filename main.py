import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


column_names = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH', 'wind', 'rain', 'area'] 
dataset = pd.read_csv('forestfires.csv', names=column_names)

print(dataset.head())

dataset['Log-area'] = np.log10(dataset['area']+1)
print(dataset.describe())

# for i in dataset.describe().columns[:-2]:
#     dataset.plot.scatter(i, 'Log-area', grid=True)

# plt.show()

### Univariate plots ###
#box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(5,3), sharex=False, sharey=False)
# #histogram
# dataset.hist()

# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# # We can set the number of bins with the `bins` kwarg

# axs[0].hist(dataset['area'], bins=20)
# plt.title("Area")
# axs[1].hist(dataset['Log-area'], bins=20)
# plt.title("LogArea")
# plt.show()

# ### Multivariate plots ###
# scatter_matrix(dataset)

enc = LabelEncoder()
dataset['month'] = enc.fit_transform(dataset['month'])
dataset['day'] = enc.fit_transform(dataset['month'])

test_size = 0.2

X_data=dataset.drop(['area', 'Log-area'], axis=1)
y_data=dataset['Log-area']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size)

#optimizing function. Regression error characteristic estimation
def rec(m,n,tol):
    if type(m)!='numpy.ndarray':
        m=np.array(m)
    if type(n)!='numpy.ndarray':
        n=np.array(n)
    l=m.size
    percent = 0
    for i in range(l):
        if np.abs(10**m[i]-10**n[i])<=tol:
            percent+=1
    return 100*(percent/l)
tol_max = 20

##### SVR ######
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
scaler = StandardScaler()
# Parameter grid for the Grid Search
param_grid = {'C': [0.01], 'epsilon': [10,1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
grid_SVR = GridSearchCV(SVR(),param_grid,refit=True,verbose=0,cv=5)
grid_SVR.fit(scaler.fit_transform(X_train), scaler.fit_transform(y_train.values.reshape(-1, 1)))
print("Best parameters obtained by Grid Search:",grid_SVR.best_params_)

a=grid_SVR.predict(X_test)
rmse_svr = np.sqrt(np.mean((y_test-a)**2))
mae_svr = np.mean(np.abs(y_test-a))
print("RMSE for Support Vector Regression:", rmse_svr)

plt.figure()
plt.title("Error vs Area Burned for SVR")
plt.xlabel("Actual area burned")
plt.ylabel("Error")
plt.scatter(10**(y_test), 10**(a)-10**(y_test))
# plt.axis('equal')
# plt.axis('square')
plt.grid(True)

plt.figure()
plt.title("Histogram of prediction errors for SVR")
plt.xlabel("Prediction error (ha)")
plt.grid(True)
plt.hist(10**(a.reshape(a.size,))-10**(y_test), bins=50)

rec_SVR=[]
for i in range(tol_max):
    rec_SVR.append(rec(a, y_test, i))

plt.figure(figsize=(5,5))
plt.title("REC curve for the Support Vector Regressor\n",fontsize=15)
plt.xlabel("Absolute error (tolerance) in prediction ($ha$)")
plt.ylabel("Percentage of correct prediction")
plt.xticks([i*5 for i in range(tol_max+1)])
plt.ylim(-10,100)
plt.yticks([i*20 for i in range(6)])
plt.grid(True)
plt.plot(range(tol_max),rec_SVR)
plt.show()
###Decision Tree Regressor####
from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor(max_depth=10, criterion="mae")
tree_model.fit(scaler.fit_transform(X_train), scaler.fit_transform(y_train.values.reshape(-1, 1)))

a=tree_model.predict(X_test)
rmse_dt = np.sqrt(np.mean((y_test-a)**2))
mae_dt = np.mean(np.abs(y_test-a))
print("RMSE for Decision Tree:", rmse_dt)

plt.figure()
plt.title("Error vs Area Burned for Decision Tree")
plt.xlabel("Actual area burned")
plt.ylabel("Error")
plt.grid(True)
plt.scatter(10**(y_test),10**(a)-10**(y_test))

plt.figure()
plt.title("Histogram of prediction errors for Decision Tree\n",fontsize=18)
plt.xlabel("Prediction error ($ha$)",fontsize=14)
plt.grid(True)
plt.hist(10**(a.reshape(a.size,))-10**(y_test),bins=50)

rec_DT=[]
for i in range(tol_max):
    rec_DT.append(rec(a,y_test,i))

plt.figure(figsize=(5,5))
plt.title("REC curve for the single Decision Tree\n",fontsize=15)
plt.xlabel("Absolute error (tolerance) in prediction ($ha$)")
plt.ylabel("Percentage of correct prediction")
plt.xticks([i for i in range(0,tol_max+1,5)])
plt.ylim(-10,100)
plt.yticks([i*20 for i in range(6)])
plt.grid(True)
plt.plot(range(tol_max),rec_DT)
plt.show()
#####Random Forest Regressor#####
from sklearn.ensemble import RandomForestRegressor
param_grid = {'max_depth': [5,10,15,20,50], 'max_leaf_nodes': [2,5,10], 'min_samples_leaf': [2,5,10],
             'min_samples_split':[2,5,10]}
grid_RF = GridSearchCV(RandomForestRegressor(),param_grid,refit=True,verbose=0,cv=5)
grid_RF.fit(X_train,y_train)
print("Best parameters obtained by Grid Search:",grid_RF.best_params_)
a=grid_RF.predict(X_test)
rmse_rf=np.sqrt(np.mean((y_test-a)**2))
mae_rf = np.mean(np.abs(y_test-a))
print("RMSE for Random Forest:",rmse_rf)

plt.figure()
plt.title("Error vs Area Burned for Random Forest")
plt.xlabel("Actual area burned")
plt.ylabel("Error")
plt.grid(True)
plt.scatter(10**(y_test),10**(a)-10**(y_test))

plt.figure()
plt.title("Histogram of prediction errors for Random Forest\n",fontsize=18)
plt.xlabel("Prediction error ($ha$)",fontsize=14)
plt.grid(True)
plt.hist(10**(a.reshape(a.size,))-10**(y_test),bins=50)

rec_RF=[]
for i in range(tol_max):
    rec_RF.append(rec(a,y_test,i))

plt.figure(figsize=(5,5))
plt.title("REC curve for the Random Forest\n",fontsize=15)
plt.xlabel("Absolute error (tolerance) in prediction ($ha$)")
plt.ylabel("Percentage of correct prediction")
plt.xticks([i for i in range(0,tol_max+1,5)])
plt.ylim(-10,100)
plt.yticks([i*20 for i in range(6)])
plt.grid(True)
plt.plot(range(tol_max),rec_RF)
plt.show()

from keras.models import Sequential
import keras.optimizers as opti
from keras.layers import Dense, Activation,Dropout
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras

# model = keras.Sequential([
#         layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X_train.keys())]),
#         layers.Dense(64, activation=tf.nn.relu),
#         layers.Dense(1)
#     ])

model = Sequential()
model.add(Dense(100, input_dim=12))
model.add(Activation('selu'))
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Activation('selu'))
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dense(1))
model.summary()
learning_rate=0.001
optimizer = opti.RMSprop(lr=learning_rate)
model.compile(optimizer=optimizer,loss='mse')

# optimizer = tf.keras.optimizers.RMSprop(0.001)
# model.compile(loss='mean_squared_error', 
#     optimizer = optimizer,
#     metrics=['mean_absolute_error', 'mean_squared_error'])
data=X_train
target = y_train
model.fit(data, target, epochs=200, batch_size=10,verbose=0)

a=model.predict(X_test)
rmse_dnn = np.sqrt(np.mean((y_test-a.reshape(a.size,))**2))
mae_dnn = np.mean(np.abs(y_test-a.reshape(a.size,)))
print("RMSE for Deep Network:", rmse_dnn)

plt.figure()
plt.title("Error vs Area Burned for DNN")
plt.xlabel("Actual area burned")
plt.ylabel("Error")
plt.grid(True)
plt.scatter(10**(y_test),10**(a.reshape(a.size,))-10**(y_test))

plt.figure()
plt.title("Histogram of prediction errors for DNN\n",fontsize=18)
plt.xlabel("Prediction error ($ha$)",fontsize=14)
plt.grid(True)
plt.hist(10**(a.reshape(a.size,))-10**(y_test),bins=50)

rec_NN=[]
for i in range(tol_max):
    rec_NN.append(rec(a,y_test,i))

plt.figure(figsize=(5,5))
plt.title("REC curve for the Deep Network\n",fontsize=15)
plt.xlabel("Absolute error (tolerance) in prediction ($ha$)")
plt.ylabel("Percentage of correct prediction")
plt.xticks([i for i in range(0,tol_max+1,5)])
plt.ylim(-10,100)
plt.yticks([i*20 for i in range(6)])
plt.grid(True)
plt.plot(range(tol_max),rec_NN)

plt.show()


###Overall Summary / Comparing Models

plt.figure(figsize=(10,8))
plt.title("REC curve for various models\n",fontsize=20)
plt.xlabel("Absolute error (tolerance) in prediction ($ha$)",fontsize=15)
plt.ylabel("Percentage of correct prediction",fontsize=15)
plt.xticks([i for i in range(0,tol_max+1,1)],fontsize=13)
plt.ylim(-10,100)
plt.xlim(-2,tol_max)
plt.yticks([i*20 for i in range(6)],fontsize=18)
plt.grid(True)
plt.plot(range(tol_max), rec_SVR,'--',lw=3)
plt.plot(range(tol_max),rec_DT,'*-',lw=3)
plt.plot(range(tol_max),rec_RF,'o-',lw=3)
plt.plot(range(tol_max),rec_NN,'k-',lw=3)
plt.legend(['SVR','Decision Tree','Random Forest','Deep NN'],fontsize=13)

print("RMSE for SVR: ", rmse_svr)
print("RMSE for Decisions Tree: ", rmse_dt)
print("RMSE for Random Forest: ", rmse_rf)
print("RMSE for DNN: ", rmse_dnn)
print("MAE for SVR: ", mae_svr)
print("MAE for Decisions Tree: ", mae_dt)
print("MAE for Random Forest: ", mae_rf)
print("MAE for DNN: ", mae_dnn)

models = ('SVR', 'Decision Tree', 'Random Forest', 'DNN')
rmse_arr = [rmse_svr, rmse_dt, rmse_rf, rmse_dnn]
y_pos = np.arange(len(models))
plt.figure()
plt.bar(y_pos, rmse_arr, align='center', alpha=0.5)
plt.xticks(y_pos, models)
plt.ylabel('Root Squared Mean Error')
plt.title('Comparing RMSE for Different Models')

mae_arr = [mae_svr, mae_dt, mae_rf, mae_dnn]
y_pos = np.arange(len(models))
plt.figure()
plt.bar(y_pos, mae_arr, align='center', alpha=0.5)
plt.xticks(y_pos, models)
plt.ylabel('Mean Absolute Error')
plt.title('Comparing MAE for Different Models')
plt.show()