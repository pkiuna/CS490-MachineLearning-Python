

import matplotlib.pyplot as plt  # plotting library
import scipy
import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import itertools
import random
from random import shuffle
from PIL import Image
from scipy import ndimage
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

train_directory = Path('../input/training/training/')
test_directory = Path('../input/validation/validation/')

#label
cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']
labels = pd.read_csv("../input/monkey_labels.txt", names=cols, skiprows=1)
labels

labels = labels['Common Name']
labels

def image_show(num_image,label):
    for i in range(num_image):
        image_directory = Path('../input/training/training/' + label)
        #print(image_Directory)
        image_file = random.choice(os.listdir(image_directory))
        #print(imgfile)
        img = cv2.imread('../input/training/training/'+ label +'/'+ imgfile)

        #print(img.shape and label
        plt.figure(i)
        plt.imshow(img)
        plt.title(image_file)
    plt.show()

print(labels[4])
image_show(3,'n4')
#Training generator
train_dataGenerator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.5,
        height_shift_range=0.5,
        shear_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = train_dataGenerator.flow_from_directory(train_directory,
                                                    target_size=(height,width),
                                                    batch_size=batch_size,
                                                    seed=seed,
                                                    shuffle=True)
#test Generator
test_dataGenerator = ImageDataGenerator(rescale=1./255)
validation_generator = test_dataGenerator.flow_from_directory(test_dir,
                                                  target_size=(height,width),
                                                  batch_size=batch_size,
                                                  seed=seed,
                                                  shuffle=False)
train_number = train_generator.samples
validation_number = validation_generator.samples

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()
history = model.fit_generator(train_generator,
                              steps_per_epoch= train_num // batch_size,
                              epochs=epochs,
                              validation_data=train_generator,
                              validation_steps= validation_num // batch_size,
                              callbacks=callbacks_list,
                              verbose = 1
                             )
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracies')
plt.plot(epochs, acc, 'green', label='Training accuracy')
plt.plot(epochs, val_acc, 'purple', label='Validation accuracy')
plt.legend()

plt.figure()
plt.title('Training and validation losses')
plt.plot(epochs, loss, 'blue', label='Training loss')
plt.plot(epochs, val_loss, 'white', label='Validation loss')

plt.legend()
plt.show()
plt.ylabel('True label')
plt.xlabel("Predicted label")

from keras.models import load_model
trained_Model = load_model
Y_pred = trained_Model.predict_generator(validation_generator, validation_number)

confusion_mtx = confusion_matrix(y_true=validation_generator.classes,y_pred=Y_pred_classes)
#plot confusion matrix
plot_confusion_matrix(confusion_mtx, normalize=True, target_names=labels)

print(metrics.classification_report(validation_generator.classes, Y_pred_classes,target_names=labels))
test_list = os.listdir("../input/test-monkeys/")
test_list.sort()
print(test_list)
model_test = load_model(filepath)















