import numpy as np
from keras.layers import Dense, Activation, Input, Concatenate
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model

# importing the dataset
#dataset = np.genfromtxt("dati.dat",delimiter=" ")
#x = dataset[:, :-1]
#y = dataset[:, -1]

import pylab
x,y,somma = pylab.loadtxt("dati.dat",unpack=True)

# creo una matrice per passare un array con 2 argomenti in input
inputs = np.zeros((len(x),2))
for count in range (len(x)):
    inputs[count][0] = x[count]
    inputs[count][1] = y[count]

# inizializzazione ANN
model = Sequential()

# input
model.add(Dense(units=2))

# hidden layer
model.add(Dense(units=20,activation='relu',use_bias=True))
model.add(Dense(units=20,activation='relu',use_bias=True))
model.add(Dense(units=20,activation='relu',use_bias=True))
model.add(Dense(units=20,activation='relu',use_bias=True))
model.add(Dense(units=20,activation='relu',use_bias=True))
model.add(Dense(units=20,activation='relu',use_bias=True))
model.add(Dense(units=20,activation='relu',use_bias=True))
model.add(Dense(units=20,activation='relu',use_bias=True))
model.add(Dense(units=20,activation='relu',use_bias=True))
model.add(Dense(units=20,activation='relu',use_bias=True))

# output
model.add(Dense(units=1))

# compiling the ANN (per problemi di regressione)
model.compile(optimizer='Adam',loss='mean_squared_error')

# per problemi di classificazione
#model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

# training
model.fit(inputs,somma,shuffle=True,batch_size=10,epochs=2000)

# test sui dati
somma_pred = model.predict(inputs)

# plot
ax = plt.axes(projection ='3d')
ax.scatter(x,y,somma,marker='o')
ax.scatter(x,y,somma_pred,marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('MOLTIPLICAZIONE')
plt.title('Moltiplicatore')
#plt.legend()
plt.show()

# salva su file il modello creato (HDF5 file)
model.save('modello.h5')

