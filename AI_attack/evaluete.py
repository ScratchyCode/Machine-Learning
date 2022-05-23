import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # da scegliere tra {'0', '1', '2'}
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.nasnet import preprocess_input
from keras.applications.nasnet import decode_predictions
from keras.applications.nasnet import NASNetLarge

filename = input("Inserisci il nome della foto: ")

print("Preprocessing dati...")
# carica la foto
image = load_img(filename,target_size=(331,331))

# immagine -> array
image = img_to_array(image)

# reshape (batch)
image = np.array([image])
#image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))

# preprocessing
image = preprocess_input(image)

# carico la cnn
print("Caricamento rete neurale...")
model = NASNetLarge(weights='imagenet')


#model.summary()
#input("Premi INVIO per continuare...")

# predict su tutte le classi in output
print("Running...")
prob = model.predict(image)

# probabilità -> label
label = decode_predictions(prob)

# i label più probabili
label1 = label[0][0]
label2 = label[0][1]
label3 = label[0][2]
label4 = label[0][3]
label5 = label[0][4]

# print
print('\nProbabilità:')
print('%s (%.2f%%)' % (label1[1],label1[2]*100))
print('%s (%.2f%%)' % (label2[1],label2[2]*100))
print('%s (%.2f%%)' % (label3[1],label3[2]*100))
print('%s (%.2f%%)' % (label4[1],label4[2]*100))
print('%s (%.2f%%)' % (label5[1],label5[2]*100))

