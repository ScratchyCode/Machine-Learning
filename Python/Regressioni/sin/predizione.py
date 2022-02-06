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
x,y = pylab.loadtxt("test.dat",unpack=True)

#carica un modello precedentemente salvato su file
model = load_model('modello.h5')

# test sui dati
y_pred = model.predict(x)

# plot
plt.scatter(x,y,marker='o',s=5,label='Dati')
plt.scatter(x,y_pred,marker='o',s=5,label='Predizioni')
plt.title('Regressione')
plt.legend()
plt.show()
