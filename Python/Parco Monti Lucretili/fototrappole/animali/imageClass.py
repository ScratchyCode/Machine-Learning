# Coded by Pietro Squilla
# Codice di sempio con dataset di Keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

#####################
### PREPROCESSING ###
#####################

# importiamo il dataset in keras con le 60000 immagini
from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# look at the data
print("Tipo di dati:")
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

# get the shape of the arrays
print("Dimensioni:")
print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('x_test shape: ', x_test.shape)
print('y_test shape: ', y_test.shape)

# look the first image as an array
index = 4
x_train[index]

# now show the image
img = plt.imshow(x_train[index])

# get the image label
print('Image label: ',y_train[index])

# do the image classification
classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','shiptruck']

# print the image class
#print('The image class is: ',classification[y_train[index][0]])

# wait
plt.show()
input("Premi INVIO per continuare...")


# convert label into set of 10 number to pass at ANN
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# print the new label
print(y_train_one_hot)

# print the new label of the image selected by index
print('The one hot label is: ',y_train_one_hot[index])

# wait
input("Premi INVIO per continuare...")

###########
### CNN ###
###########

# pixel normalization
x_train = x_train/255.
x_test = x_test/255.

# create model architecture
model = Sequential()

# il primo strato del modello deve essere un layer convoluzionale che prende l'input ed applica i filtri;
# bisogna definire:
# canali/filtri;
# l'input shape per creare il primo strato;
# funzione di attivazione e padding (relu e padding='same')
# per dire che non cambiamo la dimensione delle immagini;
# input shape è un tensore con profondità 3 perchè 3 sono i canali RGB delle foto
model.add( Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)) )

model.add( Conv2D(32,(5,5),activation='tanh',input_shape=(32,32,3)) )

# normalizzo gli input diretti verso il layer successivo
# in modo che la rete crei attivazioni sempre con la stessa distribuzione
#model.add(BatchNormalization())

model.add( MaxPooling2D(pool_size=(2,2)) )
model.add(Dropout(0.2))

# aggiungiamo un altro layer convoluzionale ma con un filtro più grande,
# cosi la rete può apprendere pattern più complessi
model.add(Conv2D(32,(3,3),activation='tanh',padding='same'))

# ora il pooling layer, che scarta la maggior parte delle informazioni nell'immagine
# che non vengono reputate necessarie per la classificazione valutando 2x2 pixel, praticamente un punto dell'immagine;
# è presente anche un dropout e una normalizzazione successiva
# add a pooling layer
#model.add( MaxPooling2D(pool_size=(2,2)) )
#model.add(Dropout(0.1))
#model.add(BatchNormalization())

# another pooling layer
#model.add( MaxPooling2D(pool_size=(2,2)) )

# questa è la prima metà di una CNN ed è composta da: convoluzione, attivazione, dropout, pooling;
# può variare il numero di strati convoluzionali, rendendo la rete capace di imparare pattern complicati.
# Invece è bene non avere molti pooling layer perchè ogni strato scarta qualche dato
# ed il loro numero varia in base alle necessità e al tipo di elaborazione;
# per immagini piccole non si inseriscono più di un paio di pooling layer

# importiamo cosi i dati in un array da passare ad una ANN
model.add( Flatten() )

###########
### ANN ###
###########

# creiamo una ANN densamente connessa;
# il numero di neuroni è meglio se diminuisce ad ogni strato
# in modo che si arrivi a quello dell'output con un numero di neuroni simile.

# maxnorm può regolarizzare i dati mentre apprende, un'altra cosa che aiuta a prevenire l'overfitting.
#model.add(Dense(256,kernel_constraint=maxnorm(3)))
model.add( Dense(1000,activation='relu') )
model.add( Dropout(0.5) )

model.add( Dense(500,activation='tanh') )
model.add( Dropout(0.5) )

model.add( Dense(250,activation='tanh') )
model.add( Dropout(0.2) )

model.add( Dense(50,activation='tanh') )
model.add( Dropout(0.1) )

# creiamo lo strato di output;
# la funzione softmax attiva i neuroni solo con alta propabilità
model.add( Dense(10,activation='softmax') )

####################
### COMPILAZIONE ###
####################

# compile
loss = 'categorical_crossentropy'
optimizer = 'adam'
model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])

# printiamo un sommario della rete cosi costruita
print(model.summary())

# wait
input("Premi INVIO per continuare...")

################
### TRAINING ###
################

epochs = 100
batch = 300
score = model.fit(x_train,y_train_one_hot,batch_size=batch,validation_split=0.2,epochs=epochs)

############
### EVAL ###
############

# eval over test data
model.evaluate(x_test,y_test_one_hot)[1]
#print("Accuracy: %.2f%%" % (score[1]*100))

# wait
input("Premi INVIO per continuare...")

# visualize model accuracy
plt.plot(score.history['accuracy'])
plt.plot(score.history['accuracy'])
plt.plot('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc='upper right')
plt.show()

# visualize model loss
plt.plot(score.history['loss'])
plt.plot(score.history['loss'])
plt.plot('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc='upper left')
plt.show()

