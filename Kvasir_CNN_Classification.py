
# import pandas as pd
from tensorflow.keras.models import  Model
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
# KERAS
import keras
from keras.datasets import mnist
from keras.models import Sequential

# TENSORFLOW 
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import ( Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, LeakyReLU,Input,InputLayer ) 

# datagen = ImageDataGenerator(
#         rotation_range=40, # rotation_range is a value in degrees (0-180), a range within which to randomly rotate pictures

# width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally
#         width_shift_range=0.2,
#         height_shift_range=0.2,

# rescale is a value by which we will multiply the data before any other processing. Our original images consist in RGB coefficients in the 0-255, but such values would be too high for our models to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255. factor.
#         rescale=1./255,

# shear_range is for randomly applying shearing transformations
#         shear_range=0.2, 

# zoom_range is for randomly zooming inside pictures
#         zoom_range=0.2,

# horizontal_flip is for randomly flipping half of the images horizontally --relevant when there are no assumptions of horizontal assymetry (e.g. real-world pictures).
#         horizontal_flip=True,

# fill_mode is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.
#         fill_mode='nearest')


train_datagen = ImageDataGenerator( 
                             rescale=1./255,
                             validation_split=0.2,
                            #  shear_range=0.2,
                            #  zoom_range=0.2,
                            #  horizontal_flip=True
                             )

test_datagen = ImageDataGenerator()

batch_size = 64

# load and iterate training dataset
train_it = train_datagen.flow_from_directory( 
                                             '../dataset/Kvasir_Dataset/Kvasir_Dataset/train/', 
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             target_size=(512, 512),  
                                             )
# # load and iterate validation dataset
val_it = train_datagen.flow_from_directory(  
                                           '../dataset/Kvasir_Dataset/Kvasir_Dataset/train/',
                                           class_mode='categorical', 
                                           batch_size=batch_size,
                                           target_size=(512, 512),  
                                           )
# load and iterate test dataset
test_it = test_datagen.flow_from_directory(  
                                           '../dataset/Kvasir_Dataset/Kvasir_Dataset/test/' , 
                                           class_mode='categorical', 
                                           batch_size=batch_size,
                                           target_size=(512, 512),  
                                           )

batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))



fashion_model = Sequential()
input_shape = (512,512,3)
# first hidden layer 
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=input_shape,padding='same')) #padding value can be same or valid, with same it has padding with valid it does not add padding
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((4, 4),padding='same'))
fashion_model.add(Dropout(0.25))

# second hidden layer
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(4, 4),padding='same'))
fashion_model.add(Dropout(0.4))

# # third hidden layer
# fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='valid'))
# fashion_model.add(LeakyReLU(alpha=0.1))                  
# fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# fashion_model.add(Dropout(0.4))

# fully connected layer
fashion_model.add(Flatten())

fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))     
fashion_model.add(Dropout(0.3))

fashion_model.add(Dense(5, activation='softmax'))

# compile model
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.optimizers.Adam(),metrics=['accuracy'])

fashion_model.summary()




fashion_model.fit(
        train_it,
        steps_per_epoch= 2000 // batch_size,
        epochs=5,
        validation_data=val_it,
        validation_steps=800 // batch_size
        )


print("Evaluating...")
loss = fashion_model.evaluate(test_it)
fashion_model.save_weights('Kvasir_Weights_CNN.h5')


# Show accuracy graph of model in training and validation
plt.plot(fashion_model.history["loss"], label="train_loss")
plt.plot(fashion_model.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()