import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.metrics import classification_report
import pickle
from threading import Thread

train = pd.read_csv(r"fashion-mnist_train.csv")
test = pd.read_csv(r"fashion-mnist_test.csv")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Prepare training data Keeping them in X and y variable.
#In X we are keeping all the columns except Label because label doesnt result into an image.
X = np.array(train.drop(columns = ['label']))
#Label corresponds to Output.
y = np.array(train['label'])
# Prepare validation data
validation_x = test.drop(columns = ['label'])
validation_y = test['label']

# Split training data into train and test
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
# Convert labels to categories
# to_categorical will transform class with numbers in proper vectors for using with models.
#You can't simply train a classification model without that.
train_y_encoded = to_categorical(train_y)
test_y_encoded = to_categorical(test_y)
val_y_encoded = to_categorical(validation_y)
# Change to numpy arrays and Normalizing.
#Because the input should always be in numpy array.
train_x = np.array(train_x)/255
test_x = np.array(test_x)/255
#Output is optional, but we are doing to be safe. To keep input and output in same format.
train_y_encoded = np.array(train_y_encoded)
test_y_encoded = np.array(test_y_encoded)
validation_x = np.array(validation_x)
val_y_encoded = np.array(val_y_encoded)

# Model Definition
input_size = 784
output_size = 10
#Model object
m1 = Sequential()
#The input layer alwways have relu as activation function
m1.add(Dense(input_size, activation = 'relu'))
#Here softmax is activation because we have multiple class as output.
m1.add(Dense(output_size, activation = 'softmax'))
#This is to compile the model.
m1.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Here we are fitting the x_train and the y_train data, for model to learn.
m1.fit(train_x, train_y_encoded, epochs = 2, batch_size = 128, validation_data =(test_x,test_y_encoded),verbose = 1)
#with verbose = 1 we will be able to see every epochs. if we put 0 then no epochs will be displayed.

y_pred = m1.predict_classes(validation_x)

cr = classification_report(y_pred, validation_y,target_names = class_names, output_dict = True)

m1.save("keras_model.h5")

#Now, we check the accuracy of our model.
print('Test Accuracy:', np.round(cr['accuracy'] * 100, 2),'%')
