import wandb
from wandb.keras import WandbCallback

wandb.init(project="test-project", entity="neurals")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
# TENSORFLOW 
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import ( Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, LeakyReLU ) 

# KERAS
import keras
from keras.datasets import mnist, cifar100
from keras.models import Sequential


(train_X, train_Y), (test_X, test_Y) = cifar100.load_data()
print('Training data shape : ', train_X.shape, train_Y.shape)

print('Testing data shape : ', test_X.shape, test_Y.shape)

classes = np.unique(train_Y)
nClasses= len(classes)

print('Total number of outputs : ' , nClasses)

print('Output Classes : ' , classes)

train_X = train_X.reshape(-1, 32,32, 3)
test_X = test_X.reshape(-1, 32,32, 3)
print(train_X.shape, test_X.shape)

# Rescale the pixel values in range 0 - 1 inclusive 
train_X = train_X.astype('float32')
test_X  = test_X.astype('float32')
train_X = train_X / 255.
test_X  = test_X / 255.

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)


train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

# model structure
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 20,
  "batch_size": 32,
  "num_classes" : 100
}

# ... Define a model

# batch_size = 32
# epochs = 20


# CNN Architecture

fashion_model = Sequential()

# first hidden layer 
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(32,32,3),padding='same')) #padding value can be same or valid, with same it has padding with valid it does not add padding
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25))

# second hidden layer
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.25))

# third hidden layer
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4))

# fully connected layer
fashion_model.add(Flatten())

fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))     
fashion_model.add(Dropout(0.3))

fashion_model.add(Dense(wandb.config['num_classes'], activation='softmax'))

# compile model
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.optimizers.Adam(),metrics=['accuracy'])

fashion_model.summary()

# train model
fashion_train = fashion_model.fit(train_X, train_label, batch_size=wandb.config['batch_size'],epochs=wandb.config['epochs'],verbose=1,validation_data=(valid_X, valid_label),
callbacks=[WandbCallback()])

plt.plot(fashion_train.history["loss"], label="train_loss")
plt.plot(fashion_train.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

# evaluation
test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# correct classes predicted
predicted_classes = fashion_model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes.shape, test_Y.shape
correct = np.where(predicted_classes==test_Y)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()

# incorrect classes predicted

incorrect = np.where(predicted_classes!=test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()

# classification report
target_names = ["Class {}".format(i) for i in range(wandb.config['num_classes'])]
print(classification_report(test_Y, predicted_classes, target_names=target_names))





