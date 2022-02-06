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

# creo il dataset di addestramento
train_ds = image_dataset_from_directory(
    directory=train_dir,
    #validation_split=0.8,
    #subset="training",
    labels='inferred',
    label_mode='categorical',
    batch_size=60,
    image_size=(height,width),
    #seed=123
)

# creo il dataset di validazione
#val_ds = image_dataset_from_directory(
#    directory=train_dir,
#    validation_split=0.2,
#    subset="validation",
#    labels='inferred',
#    label_mode='categorical',
#    batch_size=10,
#    image_size=(height,width),
#    seed=456
#)

#class_names = train_ds.class_names
#print(class_names)
#num_classes = int(len(class_names)) # quante classi sono

# numero di label
num_classes = 60

######################
### PRE-PROCESSING ###
######################

# L'overfit si verifica generalmente quando c'è un piccolo numero di esempi di train.
# L'aumento dei dati adotta l'approccio di generare dati di allenamento aggiuntivi a partire del esempi esistenti
# utilizzando delle trasformazioni casuali che producono immagini verosimili in futuro.
# Ciò consente di esporre il modello a più aspetti dei dati e di generalizzarlo meglio.

# Si implementare l'aumento dei dati utilizzando i seguenti livelli pre-elaborazione Keras:
# tf.keras.layers.RandomFlip, tf.keras.layers.RandomRotation, tf.keras.layers.RandomZoom

# incremento i pochi dati a disposizione
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal",input_shape=(height,width,3)),
    layers.RandomRotation(0.01),
    layers.RandomZoom(0.1)
])

###########
### CNN ###
###########

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
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

batch = 60
epoch = 2000

history = model.fit(
    train_ds,
    #validation_data=val_ds,
    batch_size=batch,
    epochs=epoch
)


input("Premere INVIO per contunuare...")

############
### PLOT ###
############

# visualizza l'accuracy del modello
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('accuracy modello')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.grid()
plt.show()

# visualizza la loss del modello
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('loss modello')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.grid()
plt.show()

# salva su file il modello creato (HDF5 file)
model.save('ANN.h5')

# elimina la rete creata
#del ann

#carica un modello precedentemente salvato su file
#ann = load_model('ANN.h5')
