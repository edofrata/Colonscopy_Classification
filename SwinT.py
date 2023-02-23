import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import sys
sys.path.append('../')

from keras_vision_transformer import swin_layers
from keras_vision_transformer import transformer_layers


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, 100)
y_test = keras.utils.to_categorical(y_test, 100)
# # Change the labels from categorical to one-hot encoding
# train_Y_one_hot = keras.utils.to_categorical(y_train, 10)
# test_Y_one_hot = keras.utils.to_categorical(y_test, 10)

input_size = (32, 32, 3) # The image size of the MNIST
patch_size = (2, 2) # Segment 28-by-28 frames into 2-by-2 sized patches, patch contents and positions are embedded
n_labels = 100 # MNIST labels

# Dropout parameters
mlp_drop_rate = 0.01 # Droupout after each MLP layer
attn_drop_rate = 0.01 # Dropout after Swin-Attention
proj_drop_rate = 0.01 # Dropout at the end of each Swin-Attention block, i.e., after linear projections
drop_path_rate = 0.01 # Drop-path within skip-connections

# Self-attention parameters 
# (Fixed for all the blocks in this configuration, but can vary per block in larger architectures)
num_heads = 4 # Number of attention heads
embed_dim = 16 # Number of embedded dimensions
num_mlp = 128# Number of MLP nodes
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
model = keras.models.Model(inputs=[IN,], outputs=[OUT,])

model.summary()

# Compile the model
opt = keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy',])

# Training
model_train = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.1)

# evaluation
test_eval = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# Show accuracy graph of model in training and validation
plt.plot(model_train.history["loss"], label="train_loss")
plt.plot(model_train.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

# correct classes predicted
# predicted_classes = model.predict(x_test)
# predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
# predicted_classes.shape, y_test.shape
# correct = np.where(predicted_classes==y_test)[0]
# print("Found %d correct labels" % len(correct))
# for i, correct in enumerate(correct[:9]):
#     plt.subplot(3,3,i+1)
#     plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
#     plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
#     plt.tight_layout()

# # incorrect classes predicted

# incorrect = np.where(predicted_classes!=y_test)[0]
# print("Found %d incorrect labels" % len(incorrect))
# for i, incorrect in enumerate(incorrect[:9]):
#     plt.subplot(3,3,i+1)
#     plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
#     plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
#     plt.tight_layout()

# # classification report
# target_names = ["Class {}".format(i) for i in range(n_labels)]
# print(classification_report(y_test, predicted_classes, target_names=target_names))

