# Coded by Pietro Squilla
import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
#from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras import layers
from keras.models import Sequential
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model
import pylab

###############
### DATASET ###
###############

# importo il dataset
train_dir = "dataset"
height = 20
width = 20

train_ds = image_dataset_from_directory(
    directory=train_dir,
    validation_split=0.8,
    subset="training",
    labels='inferred',
    label_mode='categorical',
    batch_size=10,
    image_size=(height,width),
    seed=123
)

# creo il dataset di validazione (con pochi dati non si ottengono buoni risultati)
val_ds = image_dataset_from_directory(
    directory=train_dir,
    validation_split=0.2,
    subset="validation",
    labels='inferred',
    label_mode='categorical',
    batch_size=10,
    image_size=(height,width),
    seed=456
)

#class_names = train_ds.class_names
#print(class_names)
#num_classes = int(len(class_names)) # quante classi sono

# numero di label
num_classes = 60

######################
### PRE-PROCESSING ###
######################

# incremento i pochi dati a disposizione
#data_augmentation = keras.Sequential([
#    layers.RandomFlip("horizontal",input_shape=(height,width,3)),
#    layers.RandomRotation(0.01),
#    layers.RandomZoom(0.1)
#])

# funzione che normalizza le immagini dei dataset
def normalizza(img):
    return img/255.

# metodo che applica la funzione normalizza a tutte le immagini dei dataset
norm_train_ds = train_ds.map(normalizza)
norm_val_ds = val_ds.map(normalizza)

###########
### CNN ###
###########

model = Sequential([
    #data_augmentation,
    #layers.Rescaling(1./255),
    layers.Conv2D(16,kernel_size=(3,3),input_shape=(height,width,3),padding='same',activation='relu'),
    layers.Dropout(0.1),
    layers.AveragePooling2D(),
    #layers.MaxPooling2D(),
    layers.Conv2D(32,kernel_size=(3,3),padding='same',activation='relu'),
    layers.Dropout(0.1),
    layers.AveragePooling2D(),
    #layers.MaxPooling2D(),
    layers.Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'),
    layers.Dropout(0.1),
    layers.AveragePooling2D(),
    #layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(400,activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(300,activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(200,activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(100,activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(num_classes,activation='softmax')
])

loss = 'categorical_crossentropy'
optimizer = 'adam'
model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])

model.summary()
input("Premere INVIO per contunuare...")

#############
### TRAIN ###
#############

epoch = 2000
batch = 60

history = model.fit(
    norm_train_ds,
    validation_data=norm_val_ds,
    batch_size=batch,
    epochs=epoch
)


input("Premere INVIO per contunuare...")

# visualizza l'accuracy del modello
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('accuracy modello')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()

# visualizza la loss del modello
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss modello')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()

# salva su file il modello creato (HDF5 file)
model.save('ANN.h5')

# elimina la rete creata
#del ann

#carica un modello precedentemente salvato su file
#ann = load_model('ANN.h5')
