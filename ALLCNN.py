import tensorflow as tf
from tensorflow.keras import layers, utils
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import graphviz
import pydot
import pydotplus
import sys

"""
alexnet = tf.keras.models.Sequential([
           layers.Conv2D(filters=96, kernel_size=(11,11), strides=4, activation='relu', input_shape=(224,224,3)),
           layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same'),
           layers.Conv2D(filters=256, kernel_size=(5,5), padding='same', activation='relu'),
           layers.MaxPooling2D(pool_size=(3,3), strides=2),
           layers.Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'),
           layers.Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'),
           layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'),
           layers.MaxPooling2D(pool_size=(3,3), strides=2),
           layers.Flatten(),
           layers.Dense(units=4096, activation='relu'),
           layers.Dropout(0.5),
           layers.Dense(units=4096, activation='relu'),
           layers.Dropout(0.5),
           layers.Dense(units=1000, activation='softmax')
])

alexnet.summary()

"""
"""
## VGG-16
input = tf.keras.Input(shape=(224,224,3))
# Convolutional Block 1
x = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', name='block1_conv1')(input)
x = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', name='block1_conv2')(x)
block1_end = layers.MaxPooling2D(pool_size=2, strides=2, name='pool1')(x)
# Convolutional Block 2
x = layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='block2_conv1')(block1_end)
x = layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='block2_conv2')(x)
block2_end = layers.MaxPooling2D(pool_size=2, strides=2, name='pool2')(x)
# Convolutional Block 3
x = layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_conv1')(block2_end)
x = layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_conv2')(x)
x = layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_conv3')(x)
block3_end = layers.MaxPooling2D(pool_size=2, strides=2, name='pool3')(x)
# Convolutional Block 4
x = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_conv1')(block3_end)
x = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_conv2')(x)
x = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_conv3')(x)
block4_end = layers.MaxPooling2D(pool_size=2, strides=2, name='pool4')(x)
# Convolutional Block 5
x = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_conv1')(block4_end)
x = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_conv2')(x)
x = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_conv3')(x)
block5_end = layers.MaxPooling2D(pool_size=2, strides=2, name='pool5')(x)
# Flattening layer
x = layers.Flatten(name='flatten')(block5_end)
# Fully connected layers
x = layers.Dense(units=4096, activation='relu', name='fc1')(x)
x = layers.Dropout(0.5, name='drop1')(x)
x = layers.Dense(units=4096, activation='relu',name='fc2')(x)
x = layers.Dropout(0.5, name='drop2')(x)
x = layers.Dense(units=1000, activation='softmax',name='classification')(x)
# Create a model
vgg_16 = tf.keras.Model(inputs=input, outputs=x)
vgg_16.summary()

"""


"""
def inception_module(prev, c1, c2, c3, c4):
  # Path 1: 1x1 convolution, c1 channels
  p1 = layers.Conv2D(filters=c1, kernel_size=(1,1), activation='relu')(prev)
  # Path 2: 1x1 convolution with channel c2[0], 3x3 convolution with channel c2[1]
  p2 = layers.Conv2D(filters=c2[0], kernel_size=(1,1), activation='relu')(prev)
  p2 = layers.Conv2D(c2[1], kernel_size=(3,3), activation='relu', padding='same')(p2)
  # Path 3: 1x1 convolution with channel c3[0], 5x5 convolution with channel c3[1]
  p3 = layers.Conv2D(filters=c3[0], kernel_size=(1,1), activation='relu')(prev)
  p3 = layers.Conv2D(filters=c3[1], kernel_size=(5,5), activation='relu', padding='same')(p3)
  # Path 4: maxpooling layer with 3x3 pool size and stride of 1, 1x1 convolution with c4 channels
  p4 = layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same')(prev)
  p4 = layers.Conv2D(filters=c4, kernel_size=(1,1), activation='relu')(p4)
  # Concatenate the outputs of all 4 paths
  output = layers.Concatenate()([p1, p2, p3, p4])
  return output

input = layers.Input(shape=[224,224,3], name='Input')
output = inception_module(input, 6, (6,6), (6,6), 6)

inception = tf.keras.Model(input, output)
inception.summary()
utils.plot_model(inception)



## GoogLeNet

# Input and stem block
# Input shape: 224,224,3
# 7x7 convolution, 3x3 maxpool, 3x3 convolution, 3x3 maxpool

input = layers.Input(shape=[224,224,3])
x = layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', activation='relu')(input)
x = layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')(x)
x = layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu')(x)
x = layers.Conv2D(filters=192, kernel_size=(3,3), padding='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')(x)

### Two inception modules and maxpool
x = inception_module(x, 64, (96, 128), (16, 32), 32)
x = inception_module(x, 128, (128, 192), (32, 96), 64)
x = layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')(x)

### Five inception modules and maxpool
x = inception_module(x, 192, (96, 208), (16, 48), 64)
x = inception_module(x, 160, (112, 224), (24, 64), 64)
x = inception_module(x, 128, (128, 256), (24, 64), 64)
x = inception_module(x, 112, (144, 288), (32, 64), 64)
x = inception_module(x, 256, (160, 320), (32, 128), 128)
x = layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')(x)

### Two Inception modules and average pooling layer
x = inception_module(x, 256, (160, 320), (32, 128), 128)
x = inception_module(x, 384, (192, 384), (48, 128), 128)
x = layers.AveragePooling2D(pool_size=(7,7))(x)

### Dropout & classification head

x = layers.Dropout(0.4)(x)
output = layers.Dense(units=10000, activation='softmax')(x)

googlenet = tf.keras.Model(input, output)
googlenet.summary()
utils.plot_model(googlenet)
"""

"""
def identity_block(input_tensor, filters, strides=1):

    f1, f2, f3 = filters

    x = layers.Conv2D(filters=f1, kernel_size=(1, 1), strides=strides)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=f2, kernel_size=(3, 3), strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=strides)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, input_tensor])
    output_tensor = layers.ReLU()(x)

    return output_tensor

input = layers.Input(shape=(224, 224, 3))
x = layers.Conv2D(filters=256, kernel_size=(7,7), strides=2, padding='same')(input)

output = identity_block(x, (64, 64, 256))
plot_model(tf.keras.Model(input, output), show_shapes=True)

def projection_block(input_tensor, filters, strides=2):

  f1, f2, f3 = filters
  x = layers.Conv2D(filters=f1, kernel_size=(1,1), strides=strides)(input_tensor)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)

  x = layers.Conv2D(filters=f2, kernel_size=(3,3), strides=1, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)

  x = layers.Conv2D(filters=f3, kernel_size=(1,1), strides=1)(x)
  x = layers.BatchNormalization()(x)

  # 1x1 conv projection shortcut
  shortcut = layers.Conv2D(filters=f3, kernel_size=(1,1), strides=strides)(input_tensor)
  shortcut = layers.BatchNormalization()(shortcut)

  x = layers.Add()([x, shortcut])
  output_tensor = layers.ReLU()(x)

  return output_tensor

input = layers.Input(shape=(224, 224, 3))

output = projection_block(input, (64, 64, 256))
plot_model(tf.keras.Model(input, output), show_shapes=True)

input = layers.Input(shape=(224, 224, 3))

x = layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same')(input)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

x = projection_block(x, (64, 64, 256))
x = identity_block(x, (64, 64, 256))
x = identity_block(x, (64, 64, 256))

x = projection_block(x, (128, 128, 512))
x = identity_block(x, (128, 128, 512))
x = identity_block(x, (128, 128, 512))
x = identity_block(x, (128, 128, 512))

x = projection_block(x, (256, 256, 1024))
x = identity_block(x, (256, 256, 1024))
x = identity_block(x, (256, 256, 1024))
x = identity_block(x, (256, 256, 1024))
x = identity_block(x, (256, 256, 1024))
x = identity_block(x, (256, 256, 1024))

x = projection_block(x, (512, 512, 2048))
x = identity_block(x, (512, 512, 2048))
x = identity_block(x, (512, 512, 2048))

x = layers.GlobalAvgPool2D()(x)
x = layers.Dense(1000, activation='softmax')(x)

resnet50 = tf.keras.Model(input, x, name='ResNet-50')

resnet50.summary()

"""
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def identity_block(input, dim):
    shortcut = input

    x = layers.Conv2D(filters=dim, kernel_size=1)(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=dim, kernel_size=3, groups=32, padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=dim * 2, kernel_size=1)(x)
    x = layers.BatchNormalization()(x)

    output = layers.Add()([shortcut, x])
    output = layers.Activation('relu')(output)

    return output


def projection_block(input, dim, strides=1):
    shortcut = layers.Conv2D(filters=2 * dim, kernel_size=1, strides=strides)(input)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters=dim, kernel_size=1)(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=dim, kernel_size=3, strides=strides, groups=32, padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=dim * 2, kernel_size=1)(x)
    x = layers.BatchNormalization()(x)

    output = layers.Add()([shortcut, x])
    output = layers.Activation('relu')(output)

    return output


input = layers.Input(shape=(224, 224, 3))

# stem
x = layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same')(input)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

# Stage 1, 3 blocks
x = projection_block(x, 128)

for _ in range(2):
  x = identity_block(x, 128)

# Stage 2, 4 blocks
x = projection_block(x, 256, 2)
for _ in range(4):
  x = identity_block(x, 256)

# Stage 3, 6 blocks
x = projection_block(x, 512, 2)
for _ in range(6):
  x = identity_block(x, 512)

# Stage 4, 3 blocks
x = projection_block(x, 1024, 2)
for _ in range(3):
  x = identity_block(x, 1024)

# classification head
x = layers.GlobalAveragePooling2D()(x)
output = layers.Dense(units=1000, activation='softmax')(x)


resnext = keras.Model(input, output, name='ResNeXt-50')

resnext.summary()
"""
"""
def entry_flow(input):
  x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, padding='same')(input)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  x = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)

  shortcut = layers.Conv2D(filters=128, kernel_size=(1,1), strides=2, padding='same')(x)
  shortcut = layers.BatchNormalization()(shortcut)

  x = layers.SeparableConv2D(filters=128, kernel_size=(3,3), padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  x = layers.SeparableConv2D(filters=128, kernel_size=(3,3), padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

  x = layers.Add()([x, shortcut])

  shortcut = layers.Conv2D(filters=256, kernel_size=(1,1), strides=2, padding='same')(x)
  shortcut = layers.BatchNormalization()(shortcut)

  x = layers.ReLU()(x)
  x = layers.SeparableConv2D(filters=256, kernel_size=(3,3), padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  x = layers.SeparableConv2D(filters=256, kernel_size=(3,3), padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

  x = layers.Add()([x, shortcut])

  shortcut = layers.Conv2D(filters=728, kernel_size=(1,1), strides=2, padding='same')(x)
  shortcut = layers.BatchNormalization()(shortcut)

  x = layers.ReLU()(x)
  x = layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  x = layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

  output_tensor = layers.Add()([x, shortcut])

  return output_tensor

def middle_flow(input_tensor):

  shortcut = input_tensor

  x = layers.ReLU()(input_tensor)
  x = layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same')(x)
  x = layers.BatchNormalization()(x)

  x = layers.ReLU()(x)
  x = layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same')(x)
  x = layers.BatchNormalization()(x)

  x = layers.ReLU()(x)
  x = layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same')(x)
  x = layers.BatchNormalization()(x)

  x = layers.Add()([x, shortcut])

  return x

def exit_flow(input_tensor):

  shortcut = layers.Conv2D(filters=1024, kernel_size=(1,1), strides=2, padding='same')(input_tensor)
  shortcut = layers.BatchNormalization()(shortcut)

  x = layers.ReLU()(input_tensor)
  x = layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  x = layers.SeparableConv2D(filters=1024, kernel_size=(3,3), padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
  x = layers.Add()([x, shortcut])

  x = layers.SeparableConv2D(filters=1536, kernel_size=(3,3), padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  x = layers.SeparableConv2D(filters=2048, kernel_size=(3,3), padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)

  x = layers.GlobalAveragePooling2D()(x)
  classifier = layers.Dense(1000, activation='softmax')(x)

  return classifier

input = layers.Input(shape=(299,299,3))

x = entry_flow(input)

#middle flow is repeated 8 times
for _ in range(8):
  x = middle_flow(x)

output = exit_flow(x)

xception = tf.keras.Model(input, output)

xception.summary()

"""
"""

def dense_block(input_tensor, k, block_reps):
  for _ in range(block_reps):

    x = layers.BatchNormalization()(input_tensor)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=4*k, kernel_size=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=k, kernel_size=3, padding='same')(x)

    output_tensor = layers.Concatenate()([input_tensor, x])

    return output_tensor
    
def transition_layers(input_tensor, theta=0.5):
  filters = input_tensor.shape[-1] * theta

  x = layers.BatchNormalization()(input_tensor)
  x = layers.ReLU()(x)
  x = layers.Conv2D(filters=filters, kernel_size=1)(x)
  output_tensor = layers.AveragePooling2D(pool_size=2, strides=2)(x)

  return output_tensor

k = 32 #growth rate

input = layers.Input(shape=(224,224,3))
x = layers.BatchNormalization()(input)
x = layers.ReLU()(x)
x = layers.Conv2D(filters=2*k, kernel_size=7, strides=2)(x)
x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

x = dense_block(x, 32, 6)
x = transition_layers(x)

x = dense_block(x, 32, 12)
x = transition_layers(x)

x = dense_block(x, 32, 32)
x = transition_layers(x)

x = dense_block(x, 32, 32)

x = layers.GlobalAveragePooling2D()(x)
output = layers.Dense(1000, activation='softmax')(x)

densenet = tf.keras.Model(input, output)

densenet.summary()
"""
"""

from tensorflow import keras
from tensorflow.keras import layers

def depthwise_block(input, filters, strides, padding='same'):

    x = layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding=padding)(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x

input = layers.Input(shape=(224, 224, 3))

#input downsampling stem
x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(input)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# depthwise separable convolutions blocks
x = depthwise_block(x, filters=32, strides=1)
x = depthwise_block(x, filters=64, strides=2)
x = depthwise_block(x, filters=128, strides=1)
x = depthwise_block(x, filters=128, strides=2)
x = depthwise_block(x, filters=256, strides=1)
x = depthwise_block(x, filters=256, strides=2)

# 5 repeated depthwise seperable conv blocks
for _ in range(5):
  x = depthwise_block(x, filters=512, strides=1)

x = depthwise_block(x, filters=1024, strides=2)
x = depthwise_block(x, filters=1024, strides=1)

# classification head
x = layers.GlobalAveragePooling2D()(x)
output = layers.Dense(units=1000, activation='softmax')(x)

mobilenet = keras.Model(input, output)
mobilenet.summary()

"""
"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import math

def mbconv_block(input, filters_in, filters_out, kernel_size=3, strides=1, exp_ratio=6, se_ratio=0.25):

  filters = filters_in * exp_ratio
  if exp_ratio != 1:
    x = layers.Conv2D(filters, kernel_size=1, padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = keras.activations.swish(x)

  else:
    x = input

  # Depthwise convolution
  x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = keras.activations.swish(x)
  # Squeeze and excitation
  if se_ratio > 0 and se_ratio <= 1:
    filters_se = max(1, int(filters_in * se_ratio)) #max with 1 to make sure filters are not less than 1
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Conv2D(filters_se, kernel_size=1, padding='same', activation='swish')(se)
    se = layers.Conv2D(filters, kernel_size=1, padding='same', activation='sigmoid')(se)
    x = layers.multiply([x, se])

  x = layers.Conv2D(filters_out, kernel_size=1, padding='same')(x)
  x = layers.BatchNormalization()(x)

  # Add identity shortcut if strides=2 and in & filters are same
  if strides == 1 and filters_in == filters_out:
    x = layers.add([x, input])

  return x


def scale_number_of_blocks(block_repeats, depth_coefficient=1):


    scaled_blocks = int(math.ceil(block_repeats * depth_coefficient))

    return scaled_blocks

def scale_width(filters, width_coefficient=1, depth_divisor=8):

  filters *= width_coefficient
  new_filters = (filters + depth_divisor / 2) // depth_divisor * depth_divisor
  new_filters = max(depth_divisor, new_filters)

  # make sure that scaled filters down does not go down by more than 10%
  if new_filters < 0.9 * filters:
    new_filters += depth_divisor

  return int(new_filters)

# Setting some hyperparameters for EfficientNet-B0

input_shape = (224, 224, 3)

# If using the scaling utility functions, use the following hyperparamaters for EfficientNet-B0
# depth_coefficient = 1.0
# width_coefficient = 1.0
# depth_divisor = 8

# The stem of network
input = layers.Input(input_shape)
x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(input)
x = layers.BatchNormalization()(x)
x = keras.activations.swish(x)

# MBConv blocks
# Block 1: input filters=32, output filters=16, kernel size=3, block repeats=1
x = mbconv_block(x, filters_in=32, filters_out=16, kernel_size=3, strides=1, exp_ratio=1)

# Block 2: input filters=16, output filters=24, kernel size=3, strides=2, block repeats=2
# the first block of every stage has stride of 1
x = mbconv_block(x, filters_in=16, filters_out=24, kernel_size=3, strides=1, exp_ratio=6)
x = mbconv_block(x, filters_in=16, filters_out=24, kernel_size=3, strides=2, exp_ratio=6)

# Block 3: input filters=24, output filters=40, kernel size=5, strides=2, block repeats=2
x = mbconv_block(x, filters_in=24, filters_out=40, kernel_size=5, strides=1, exp_ratio=6)
x = mbconv_block(x, filters_in=24, filters_out=40, kernel_size=5, strides=2, exp_ratio=6)

# Block 4: input filters=40, output filters=80, kernel size=3, strides=2, block repeats=3
x = mbconv_block(x, filters_in=40, filters_out=80, kernel_size=3, strides=1, exp_ratio=6)
x = mbconv_block(x, filters_in=40, filters_out=80, kernel_size=3, strides=2, exp_ratio=6)
x = mbconv_block(x, filters_in=40, filters_out=80, kernel_size=3, strides=2, exp_ratio=6)
x = mbconv_block(x, filters_in=40, filters_out=80, kernel_size=3, strides=2, exp_ratio=6)

# Block 5: input filters=80, output filters=112, kernel size=5, strides=1, block repeats=3
x = mbconv_block(x, filters_in=80, filters_out=112, kernel_size=5, strides=1, exp_ratio=6)
x = mbconv_block(x, filters_in=80, filters_out=112, kernel_size=5, strides=1, exp_ratio=6)
x = mbconv_block(x, filters_in=80, filters_out=112, kernel_size=5, strides=1, exp_ratio=6)

# Block 6: input filters=112, output filters=192, kernel size=5, strides=2, block repeats=4
x = mbconv_block(x, filters_in=112, filters_out=192, kernel_size=5, strides=1, exp_ratio=6)
x = mbconv_block(x, filters_in=112, filters_out=192, kernel_size=5, strides=2, exp_ratio=6)
x = mbconv_block(x, filters_in=112, filters_out=192, kernel_size=5, strides=2, exp_ratio=6)
x = mbconv_block(x, filters_in=112, filters_out=192, kernel_size=5, strides=2, exp_ratio=6)

# Block 7: input filters=192, output filters=320, kernel size=3, strides = 1, block repeats=1
x = mbconv_block(x, filters_in=192, filters_out=320, kernel_size=3, strides=1, exp_ratio=6)

# Classification head
x = layers.Conv2D(filters=1280, kernel_size=1, padding='same')(x)
x = layers.BatchNormalization()(x)
x = keras.activations.swish(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(units=1000, activation='softmax')(x)

efficientnet_b0 = keras.Model(inputs=input, outputs=output)

efficientnet_b0.summary()
eff_b0 = keras.applications.EfficientNetB0(include_top=True, input_shape=(224, 224, 3))
eff_b0.summary()
"""

# "depths": [2, 6, 15, 2],
# "widths": [96, 192, 432, 1008],
# "group_width": 48,
# "input resolution": 224,
# "block_type": "X(no squeeze and excitation)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

def stem(input):

  x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(input)
  x = layers.BatchNormalization(epsilon=1e-5)(x)
  x = layers.ReLU()(x)

  return x


def conv_block(input, filters_out, group_width=48, strides=1):

    groups = filters_out // group_width

    x = layers.Conv2D(filters=filters_out, kernel_size=1, strides=1)(input)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=filters_out, kernel_size=3, strides=strides, groups=groups, padding='same')(x)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=filters_out, kernel_size=1, strides=1)(x)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = layers.ReLU()(x)

    if strides == 1:
        shortcut = input
        x = layers.Add()([x, shortcut])

    if strides == 2:
        shortcut = layers.Conv2D(filters=filters_out, kernel_size=1, strides=strides)(input)
        shortcut = layers.BatchNormalization(epsilon=1e-5)(shortcut)
        x = layers.Add()([x, shortcut])

    return x

# Building RegNetX-032

input_shape = (224, 224, 3)
num_classes = 1000 # per imagenet

input = layers.Input(shape=input_shape)

# Stem
x = stem(input)

# RegNetX032 blocks
# Each stage has number of blocks n where the first block of each stage has stride of 2 in grouped convolution

# stage 1: 2 blocks, 96 channels(filters_out)
x = conv_block(x, 96, group_width=48, strides=2)
x = conv_block(x, 96, group_width=48, strides=1)

# stage 2: 6 blocks, 192 channels(filters_out)
x = conv_block(x, 192, group_width=48, strides=2)
for _ in range(1, 6):
  x = conv_block(x, 192, group_width=48, strides=1)

# stage 3: 15 blocks, 432 channels(filters_out)
x = conv_block(x, 432, group_width=48, strides=2)
for _ in range(1, 15):
  x = conv_block(x, 432, group_width=48, strides=1)

#stage 4: 2 blocks, 1008 channels(filters_out)
x = conv_block(x, 1008, group_width=48, strides=2)
x = conv_block(x, 1008, group_width=48, strides=1)

# classification head
x = layers.GlobalAveragePooling2D()(x)
output = layers.Dense(units=num_classes, activation='softmax')(x)

#build the model
regnetx_032 = keras.Model(inputs=input, outputs=output)


regnetx_032.summary()

"""
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


def ConvMixer(h,d,k,p,n):
 S,C,A=Sequential,Conv2d,lambda x:S(x,GELU(),BatchNorm2d(h))
 R=type('',(S,),{'forward':lambda s,x:s[0](x)+x})
 return S(A(C(3,h,p,p)),*[S(R(A(C(h,h,k,groups=h,padding=k//2))),A(C(h,h,1))) for i in range(d)],AdaptiveAvgPool2d(1),Flatten(),Linear(h,n))

def convmixer_block(input, filters, kernel_size):
  shortcut = input

  # Depthwise convolution
  x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same')(input)
  x = keras.activations.gelu(x)
  x = layers.BatchNormalization()(x)

  # Shortcut connection
  x = layers.Add()([shortcut, x])

  # Pointwise or 1x1 convolution
  x = layers.Conv2D(filters=filters, kernel_size=1, padding='same')(x)
  x = keras.activations.gelu(x)
  x = layers.BatchNormalization()(x)

  return x

# Defining some hyperparameters for ConvMixer-1536/20
input_shape = (224, 224, 3)
patch_size = 7
depth = 20
kernel_size = 9
filters = 1536
num_classes = 1000 #per imagenet

# Input and patch embedding layer
input = layers.Input(input_shape)
x = layers.Conv2D(filters=filters, kernel_size=patch_size, strides=patch_size)(input)
x = keras.activations.gelu(x)
x = layers.BatchNormalization()(x)

# ConvMixer blocks repeated depth times
for _ in range(depth):
  x = convmixer_block(x, filters, kernel_size)

# Classification head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(units=num_classes, activation='softmax')(x)

convmixer = keras.Model(inputs=input, outputs=x)

convmixer.summary()

"""
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, utils
import tensorflow_addons as tfa


def convnext_block(input, dim, drop_path=0.0):
    shortcurt = input  # shortcut connection
    x = layers.Conv2D(filters=dim, kernel_size=7, padding='same', groups=dim)(input)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(4 * dim)(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dense(dim)(x)

    # Stochastic depth
    drop_depth = tfa.layers.StochasticDepth(drop_path) if drop_path > 0.0 else layers.Activation("linear")

    output = layers.Add()([shortcurt, drop_depth(x)])

    return output

def stem(input, dim):

  x = layers.Conv2D(filters=dim, kernel_size=4, strides=4)(input)
  x = layers.LayerNormalization(epsilon=1e-6)(x)

  return x

def downsampling_layers(input, dim):

  x = layers.LayerNormalization(epsilon=1e-6)(input)
  x = layers.Conv2D(filters=dim, kernel_size=2, strides=2)(x)

  return x


def convnext_model(input_shape=(224, 224, 3), dims=[96, 192, 384, 768], num_classes=1000):
    input = layers.Input(input_shape)

    # stem
    x = stem(input, dims[0])

    # Convnext stage 1 x3, dim[0] = 96
    for _ in range(3):
        x = convnext_block(x, dims[0])

    # Downsampling layers + stage 2 x3, dim[1] = 192
    x = downsampling_layers(x, dims[1])
    for _ in range(3):
        x = convnext_block(x, dims[1])

    # Downsampling layers + stage 3 x9, dim[2] = 384
    x = downsampling_layers(x, dims[2])
    for _ in range(9):
        x = convnext_block(x, dims[2])

    # Downsampling layers + stage 4 x3, dim[3] = 768
    x = downsampling_layers(x, dims[3])
    for _ in range(3):
        x = convnext_block(x, dims[3])

    # Classification head: Global average pool + layer norm + fully connected layer
    x = layers.GlobalAvgPool2D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    output = layers.Dense(units=num_classes, activation='softmax')(x)

    model = keras.Model(input, output, name='ConvNeXt')

    return model


convnext = convnext_model()
convnext.summary()

"""