import wandb
from wandb.keras import WandbCallback

wandb.init(project="Swin-T-Edo", entity="mef")


# import pandas as pd
from tensorflow.keras.models import  Model

from keras.preprocessing.image import ImageDataGenerator


import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
from sklearn.metrics import classification_report

import sys
sys.path.append('../')

from keras_vision_transformer import swin_layers
from keras_vision_transformer import transformer_layers

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
                            # rescale=1./255,
                            # validation_split=0.2,
                            #  shear_range=0.2,
                            #  zoom_range=0.2,
                            #  horizontal_flip=True
                             )

test_datagen = ImageDataGenerator()

batch_size = 128

# load and iterate training dataset
train_it = train_datagen.flow_from_directory( 
                                             '../dataset/Kvasir_Dataset/Kvasir_Dataset/train/', 
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             target_size=(512, 512),  
                                             )
# load and iterate validation dataset
# val_it = train_datagen.flow_from_directory(  
#                                            '../dataset/Kvasir_Dataset/Kvasir_Dataset/train/',
#                                            class_mode='categorical', 
#                                            batch_size=batch_size,
#                                            target_size=(512, 512),  
#                                            )
# load and iterate test dataset
test_it = test_datagen.flow_from_directory(  
                                           '../dataset/Kvasir_Dataset/Kvasir_Dataset/test/' , 
                                           class_mode='categorical', 
                                           batch_size=batch_size,
                                           target_size=(512, 512),
                                           )

batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

input_size = (512,512,3)
patch_size = (2, 2) # 2-by-2 sized patches, patch contents and positions are embedded
n_labels = 5 # labels

# Dropout parameters
mlp_drop_rate  = 0.01 # Droupout after each MLP layer
attn_drop_rate = 0.01 # Dropout after Swin-Attention
proj_drop_rate = 0.01 # Dropout at the end of each Swin-Attention block, i.e., after linear projections
drop_path_rate = 0.01 # Drop-path within skip-connections

# Self-attention parameters 
# (Fixed for all the blocks in this configuration, but can vary per block in larger architectures)
num_heads = 4  # Number of attention heads
embed_dim = 32 # Number of embedded dimensions
num_mlp = 256 # Number of MLP nodes
qkv_bias = True # Convert embedded patches to query, key, and values with a learnable additive value
qk_scale = None # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor

# Shift-window parameters
window_size = 2 # Size of attention window (height = width)
shift_size = window_size // 2 # Size of shifting (shift_size < window_size)

num_patch_x = input_size[0]//patch_size[0]
num_patch_y = input_size[1]//patch_size[1]

# The input section
IN = Input(input_size)
X = IN

# Extract patches from the input tensor
X = transformer_layers.patch_extract(patch_size)(X)

# Embed patches to tokens
X = transformer_layers.patch_embedding(num_patch_x*num_patch_y, embed_dim)(X)


# -------------------- Swin transformers -------------------- #
# Stage 1: window-attention + Swin-attention + patch-merging
for i in range(2):
    if i % 2 == 0:
        shift_size_temp = 0
    else:
        shift_size_temp = shift_size

    X = swin_layers.SwinTransformerBlock(dim=embed_dim, num_patch=(num_patch_x, num_patch_y), num_heads=num_heads, 
                             window_size=window_size, shift_size=shift_size_temp, num_mlp=num_mlp, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             mlp_drop=mlp_drop_rate, attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, drop_path_prob=drop_path_rate, 
                             name='swin_block{}'.format(i))(X)
# Patch-merging
#    Pooling patch sequences. Half the number of patches (skip every two patches) and double the embedded dimensions
X = transformer_layers.patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(i))(X)

# ----------------------------------------------------------- #


# Convert embedded tokens (2D) to vectors (1D)
X = GlobalAveragePooling1D()(X)

# The output section
OUT = Dense(n_labels, activation='softmax')(X)

# Model configuration
SwinT_model = keras.models.Model(inputs=[IN,], outputs=[OUT,])

SwinT_model.summary()


# Compile the model
opt = keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5)
SwinT_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy',])


SwinT_model.fit(
        train_it,
        steps_per_epoch= 2000 // batch_size,
        epochs=5,
        # validation_data=val_it,
        validation_steps=800 // batch_size,
        callbacks=[WandbCallback()]
        )

print("Evaluating...")
# evaluating the model performance
SwinT_model.evaluate(test_it) 
# saving the weights of the model 
SwinT_model.save_weights('Kvasir_Weights_SwinT.h5')

