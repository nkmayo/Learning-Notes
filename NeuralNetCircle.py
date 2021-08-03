# %%
"""Simple Keras model to test for visible effects of using different activation functions"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# %%
# create points (predictors) randomly between -20 and 20
predictors = np.random.randint(-20,20,size=(10000,2))
# create the targets
target = np.array(
    # as 1 if the radius of the x, y point in X is > 16 else 0
    [np.sqrt(row[0]**2+row[1]**2)>16 for row in predictors]
    ) # .astype(int) # as an int instead of bool

X_val = np.random.randint(-20,20,size=(1000,2))
y_val = np.array(
    # as 1 if the radius of the x, y point in X is > 16 else 0
    [np.sqrt(row[0]**2+row[1]**2)>16 for row in X_val]
    ) # .astype(int) # as an int instead of bool

X_test = np.random.randint(-40,40,size=(1000,2))
y_test = np.array(
    # as 1 if the radius of the x, y point in X is > 16 else 0
    [np.sqrt(row[0]**2+row[1]**2)>16 for row in X_test]
    ) # .astype(int) # as an int instead of bool
# predictors = np.array(predictors.drop(['target'], axis=1))


ncols = predictors.shape[1]

# %%
model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(ncols,), kernel_initializer='normal'))
model.add(BatchNormalization())
# model.add(Dense(3, activation='sigmoid', kernel_initializer='normal'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print('---Pre evaluation---')
model.evaluate(predictors, target)
print('---Training---')
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
# history = model.fit(predictors, target, validation_data=(X_val, y_val), callbacks=[early_stopping],
#                    batch_size=100, epochs=200, verbose=0)
# or simply use validation_split instead of definine new validation data separately
history = model.fit(predictors, target, validation_split=0.1, callbacks=[early_stopping],
                    batch_size=100, epochs=200, verbose=0)
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0.75,1)
plt.xlabel('Epoch')
plt.legend(['Train', 'Validate'])
plt.show()

print('---Post evaluation---')
model.evaluate(X_test, y_test)

# model accuracy is good, even with only 4 nodes and given a test set outside the training range
y_pred = model.predict(X_test)

data = {'x': X_test[:,0], 'y': X_test[:,1], 't': y_test, 'p': np.reshape(y_pred, (1000,))}
data = pd.DataFrame(data)
g = sns.relplot(x='x',
            y='y',
            data=data,
            hue='p',
            )
g.fig.suptitle('Test Data Set', fontsize=20, fontweight='bold')
plt.show()

y_pred_bool = (y_pred>0.5)

cm = confusion_matrix(y_test, y_pred_bool)
print('Confusion Matrix:\n', cm)

# print('y_test: ', y_test, 'y_pred: ', y_pred)
# %%
#data = {'x': np.reshape(X_test[:,0], (1000,1)), 'y': np.reshape(X_test[:,1], (1000,1)), 't': y_pred}
#df2 = pd.DataFrame(data)
data = {'x': X_test[:,0], 'y': X_test[:,1], 't': y_test, 'p': np.reshape(y_pred, (1000,))}
df1 = pd.DataFrame(data)


# %% not what we want. this is point density
sns.kdeplot(data=data.drop('t', axis=1),
            x='x',
            y='y',
            fill=True,
            thresh=0,
            levels=100,
            cmap='mako')
plt.show()
# %% What we want, but a bit pixelated
sns.heatmap(data=data.pivot_table(index='y', columns='x', values='p'),
            #fill=True,
            #thresh=0,
            #levels=100,
            #cmap='mako')
)
plt.show()
# %% hmm takes forever because it tries to plot each different p as a new color unlike relplot
sns.histplot(data=data,
             x='x',
             y='y',
             hue='p',
             # bins=(100,100)
             )
plt.show()

# %%
