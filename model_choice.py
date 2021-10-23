from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

path = 'F:\\Python Code\\Project\\Car Price prediction\\clean_df2.csv'
df = pd.read_csv(path, index_col=0)
X_full = df
Y_full = df.price
X_full.drop('price', axis=1, inplace=True)
predictor = ['length', 'width', 'curb-weight', 'engine-size', 'horsepower', 'city-L/100km', 'highway-L/100km', 'wheel-base', 'bore', 'fuel-type-diesel', 'fuel-type-gas', 'aspiration-std', 'aspiration-turbo', 'drive-4wd', 'drive-fwd', 'drive-rwd']
#final mean_normalization
for abc in predictor:
    if X_full[abc].max()>0:
        X_full[abc] = X_full[abc]/X_full[abc].max()
X_full = X_full[predictor]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=0.2)

# linear multiveriable model
model1 = LinearRegression()

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
MAE_cross = cross_val_score(model1, X_full, Y_full, cv=5, scoring='neg_mean_absolute_error')

print('MAE: mean is %2f & std is %2f' %(-1*MAE_cross.mean(), MAE_cross.std()))
print('accuracy:', 100-100*(-1)*MAE_cross.mean()/Y_test.mean(),'%')

# Non-linear model ==> overfitting
from sklearn.preprocessing import PolynomialFeatures

print('nhap so bac:')
degree = input()
pr = PolynomialFeatures(degree=int(degree))
Z = pr.fit_transform(X_full)
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(Z, Y_full, test_size=0.2)
model2 = LinearRegression()
model2.fit(X_train1,Y_train1)
predict = model2.predict(X_test1)

#ve bieu do kiem tra phan bo predict vs truth price
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(predict,Y_test1)
plt.show()
plt.close()

#ve bieu do predict vs truth price theo khoang gia tri price
plt.figure(figsize=(width, height))
ax1 = sns.distplot(Y_test1, hist=False, color="r", label="Actual Value")
sns.distplot(predict, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()

#final model
import tensorflow as tf
from tensorflow import keras

X_full = np.array(X_full)
y_full = np.array(Y_full)
model = tf.keras.Sequential([keras.layers.Dense(64, activation='relu', input_shape=[16,]),
                             keras.layers.Dense(64, activation='relu'),
                             keras.layers.Dense(1)])
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
history = model.fit(X_full,y_full, validation_split=0.2, epochs=200, verbose=0)

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.show()
plot_loss(history)

print('DNN model result:', history.history['mean_absolute_error'][-1])
print('Regression model result:', -1*MAE_cross.mean())
