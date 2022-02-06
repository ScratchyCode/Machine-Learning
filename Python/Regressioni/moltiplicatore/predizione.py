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
    
#carica un modello precedentemente salvato su file
#model = load_model('modello.h5')
model = load_model('moltiplicatore.h5')

inputs = [[1,2],[4,5]]

moltiplicazione = model.predict(inputs)

print("\n\nOUTPUT:\n")
print(moltiplicazione)

