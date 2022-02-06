import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model
import pylab

# importing the dataset
#dataset = np.genfromtxt("dati.dat",delimiter=" ")
#x = dataset[:, :-1]
#y = dataset[:, -1]

giorni_previsione = 30

# giorni, contagiati, morti
x,y,z = pylab.loadtxt("covid.dat",unpack=True)

# inizializzazione ANN
model = Sequential()

# input
model.add(Dense(units=1))
#model.add(Dense(32,activation='relu',input_dim=1))

# hidden layer
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=25,activation='relu'))

model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=25,activation='relu'))

# output
model.add(Dense(units=1))

# compiling the ANN (per problemi di regressione)
model.compile(optimizer='adam',loss='mean_squared_error')

# per problemi di classificazione
#model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

# training
model.fit(x,y,shuffle=True,batch_size=10,epochs=10000)

# test sui dati
x_test = np.linspace(min(x),max(x) + giorni_previsione,100)
y_pred = model.predict(x_test)

# estrapolazione
var_x = 0.01
var_y = 2
alfa = np.sqrt(var_x * ((max(x)-min(x))/var_y))
estrap_err = (1/alfa) * np.sqrt(np.power(x_test,2.) - 2*x_test*np.mean(x_test) + np.power(x_test,2.))

# plot
plt.scatter(x,y,marker='o',s=5,label='Dati')
plt.scatter(x_test,y_pred,marker='o',s=5,label='Predizioni')
#plt.errorbar(x_test,y_pred,yerr=estrap_err,xerr=None)
plt.title('Regressione')
plt.legend()
plt.show()

# salva su file il modello creato (HDF5 file)
model.save('modello.h5')

# elimina la rete creata
#del model

#carica un modello precedentemente salvato su file
#model = load_model('modello.h5')
