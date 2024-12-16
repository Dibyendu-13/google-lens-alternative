import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def preprocess_images(dataset_dir, target_size=(224, 224)):
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(
        os.path.join(dataset_dir, 'train'),
        target_size=target_size,
        batch_size=32,
        class_mode='categorical'
    )
    
    test_gen = datagen.flow_from_directory(
        os.path.join(dataset_dir, 'test'),
        target_size=target_size,
        batch_size=32,
        class_mode='categorical'
    )
    
    train_images, train_labels = next(train_gen)
    test_images, test_labels = next(test_gen)
    
    return train_images, test_images, train_labels, test_labels
