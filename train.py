import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import scipy
import shutil
import numpy as np
import splitfolders
from PIL import Image
import tensorflow as tf
from keras import layers
from tensorflow import keras
from keras.applications.vgg16 import VGG16


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def make_final_model(input_shape=(224, 224, 3), learning_rate=0.1, inner_layer_size=100, droprate=0.2):
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )
    base_model.trainable = False 

    #############################

    inputs = keras.Input(shape=input_shape)
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner_layer = keras.layers.Dense(inner_layer_size, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner_layer)
    outputs = keras.layers.Dense(8)(drop)
    model = keras.Model(inputs, outputs)

    #############################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True) 

    model.compile(
        optimizer=optimizer, 
        loss=loss, 
        metrics=['accuracy']
    )

    return model

def split_the_data(output_folder):
    input_folder = 'data/natural_images'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    splitfolders.ratio(input_folder, output=output_folder, seed=SEED, ratio=(0.8, 0.0, 0.2))
    
    
def train_the_model(output_folder):
    image_size = 224
    batch_size = 32

    train_val_ds = keras.utils.image_dataset_from_directory(
        directory=f'{output_folder_no_val}/train',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(image_size, image_size)
    )

    test_ds = keras.utils.image_dataset_from_directory(
        directory=f'{output_folder_no_val}/test',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(image_size, image_size)
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_val_ds = train_val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    
    final_model = make_final_model(input_shape=(image_size, image_size, 3), learning_rate=0.01, inner_layer_size=200, droprate=0.2)
    
    final_model.fit(train_val_ds, epochs=10)
    
    return final_model

def convert_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
 
    tflite_model = converter.convert()

    # Saving the model
    with open('final_model.tflite', 'wb') as f_out:
        f_out.write(tflite_model)

output_folder_no_val = "prepared_data_no_val"
split_the_data(output_folder_no_val)
final_model = train_the_model(output_folder_no_val)
convert_model(final_model)