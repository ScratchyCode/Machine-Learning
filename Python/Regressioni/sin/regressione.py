import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model

# importing the dataset
#dataset = np.genfromtxt("dati.dat",delimiter=" ")
#x = dataset[:, :-1]
#y = dataset[:, -1]

import pylab
x,y = pylab.loadtxt("dati.dat",unpack=True)
x_test = x

# inizializzazione ANN
model = Sequential()

# input
model.add(Dense(units=1))
#model.add(Dense(32,activation='relu',input_dim=1))

# hidden layer
model.add(Dense(units=10,activation='relu'))
model.add(Dense(units=10,activation='relu'))
model.add(Dense(units=10,activation='relu'))
model.add(Dense(units=10,activation='relu'))
model.add(Dense(units=10,activation='relu'))
model.add(Dense(units=10,activation='relu'))
model.add(Dense(units=10,activation='relu'))
model.add(Dense(units=10,activation='relu'))
model.add(Dense(units=10,activation='relu'))
model.add(Dense(units=10,activation='relu'))

# output
model.add(Dense(units=1))

# compiling the ANN (per problemi di regressione)
model.compile(optimizer='adam',loss='mean_squared_error')

# per problemi di classificazione
#model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

# training
model.fit(x,y,shuffle=True,batch_size=100,epochs=1000)

# test sui dati
y_pred = model.predict(x_test)

# plot
plt.scatter(x,y,marker='o',s=5,label='Dati')
plt.scatter(x_test,y_pred,marker='o',s=5,label='Predizioni')
plt.title('Regressione')
plt.legend()
plt.show()

# salva su file il modello creato (HDF5 file)
model.save('modello.h5')

# elimina la rete creata
#del model

#carica un modello precedentemente salvato su file
#model = load_model('modello.h5')
