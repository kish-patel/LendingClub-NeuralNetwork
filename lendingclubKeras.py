# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:11:47 2020

@author: patel
"""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import itertools
import datetime



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# set random seed for reproducibility
seed = 24
np.random.seed(seed)
num_epochs = 10000
batchsize = 32
stopcriteria = EarlyStopping(monitor='val_loss', mode='min', verbose = 1, patience = 200)
checkpoint = ModelCheckpoint('best_model-noweight.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# load cleaned loan dataset
loan_df = pd.read_csv('loan_data_cleaned.csv', header = 0)

# split into input (X) and output (Y) variables
X = loan_df.loc[:, loan_df.columns != 'loan_status']
Y = loan_df['loan_status']

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Extract the numerical columns in inputs
num_columns = list(X.select_dtypes(include = ['float']).columns)
x_num = X[num_columns]
# Scale the numerical columns using MinMaxScaler
scaler = MinMaxScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x_num), columns=x_num.columns)
# Convert to numpy array
x_array = x_scaled.to_numpy()

# Extract the categorical columns in inputs
cat_columns = list(X.select_dtypes(include = ['object']).columns)
x_cat = X[cat_columns]
# Convert to dummy variables using One Hot Encoding
x_dummy = pd.get_dummies(x_cat)
# Convert to a numpy array
x_catarray = x_dummy.to_numpy()
X_result = np.column_stack((x_array, x_catarray))

# Convert Y varaible to categorical
Y_cat = Y.apply(lambda x: 1 if x == 'Default' else 0)
Y_result = Y_cat.to_numpy()

# Cleanup variables that are not needed from the workspace
del loan_df
del X
del Y
del num_columns
del x_num
del scaler
del x_scaled
del x_array
del cat_columns
del x_cat
del x_dummy
del x_catarray
del Y_cat

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start_time = datetime.datetime.now()

# Function to create model, required for KerasClassifier
def NN_model(neurons_first=512):
    # create model
    model = Sequential()
    # Add first layer - input shape is number of features/columns in X data
    model.add(Dense(neurons_first, input_dim=X_result.shape[1], activation='relu'))
    # Add dropout layer
    model.add(Dropout(0.2))
    # Add the output layer
    model.add(Dense(1, activation='sigmoid'))
    # Set the optimizer parameters
    optimizer = optimizers.SGD(learning_rate=0.001)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Create a train and test set splitting the data 70/30 into train and test respectively
X_train, X_test, y_train, y_test = train_test_split(X_result, Y_result, test_size = 0.3, random_state = 24)

# Calculate class imbalance ratio and assign appropraite class weights
np.unique(y_train, return_counts=True)
weights = {0:1,1:4.5}

# Call the model
model = NN_model()
# Print the model summary
model.summary()
# Fit the model
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batchsize, verbose =1, class_weight=weights, 
                    validation_split=0.2, callbacks=[stopcriteria, checkpoint])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Evaluate model
results = model.evaluate(X_test, y_test, batch_size=32)
# Get model probability predictions
yhat = model.predict(X_test)
# Calculate ROC AUC score
score = roc_auc_score(y_test, yhat)
print('ROC AUC: %.3f' % score)
# Get model prediction of classes
y_pred = model.predict_classes(X_test)


"""
 #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
   
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# compute the confusion matrix
confusion_mtx = confusion_matrix(y_test, y_pred) 
print(confusion_mtx)

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(1)) 


# Visualize model history
plt.figure()
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training vs. Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training vs. Validation Loss')
plt.ylabel('Loss value')
plt.xlabel('Epoch')
plt.legend(loc="upper left")

stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)