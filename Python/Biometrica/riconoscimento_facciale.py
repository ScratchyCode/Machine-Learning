# Coded da Pietro Squilla

# Questa è una dimostrazione di riconoscimento facciale sullo stream video della webcam.
# Include alcuni piccoli accorgimenti per farlo funzionare più velocemente:
#     - elabora ogni fotogramma video con risoluzione 1/4 (sebbene venga ancora visualizzato a piena risoluzione)
#     - rileva i volti solo in un fotogramma video su due.
# NB: vale a titolo di esempio; non implementa delle mitigazioni necessarie per classificare gli attacchi più frequenti alla cybersecurity

import face_recognition
import cv2
import numpy as np
import os
from PIL import Image

def converti_in_jpeg(immagine_pil):
    if immagine_pil.mode != 'RGB':
        immagine_pil = immagine_pil.convert('RGB')
    immagine_opencv = np.array(immagine_pil)
    immagine_opencv = immagine_opencv[:, :, ::-1]
    return immagine_opencv

# ottiene lo stream della webcam (quella predefinita)
video_cattura = cv2.VideoCapture(0)

# allocazione delle variabili
posizioni_volti = []
codifiche_volti = []
nomi_volti = []
processa_questo_frame = True

# carica le immagini
dir_volti = input("Inserisci il nome della directory con le immagini: ")

# verifica che la directory esiste e contiene file
print(os.listdir(dir_volti))

print("Caricamento dati...")
codifiche_volti_conosciuti = []
nomi_volti_conosciuti = []

for nome_persona in os.listdir(dir_volti):
    dir_persona = os.path.join(dir_volti, nome_persona)
    if os.path.isdir(dir_persona):  
        # verifica che la sottocartella esiste e contiene file
        print(os.listdir(dir_persona))
        for immagine_persona in os.listdir(dir_persona):
            percorso_immagine = os.path.join(dir_persona, immagine_persona)
            try:
                print("Apertura dell'immagine ", percorso_immagine, "...")
                immagine_pil = Image.open(percorso_immagine)
                print("Caricamento immagine ", immagine_persona, "...")
                immagine = converti_in_jpeg(immagine_pil)
                codifica_volto = face_recognition.face_encodings(immagine)
                if len(codifica_volto) > 0:
                    codifiche_volti_conosciuti.append(codifica_volto[0])
                    nomi_volti_conosciuti.append(nome_persona)
                else:
                    print("Nessun volto trovato nell'immagine", immagine_persona)
            except Exception as e:
                print("Errore nel caricamento dell'immagine", immagine_persona, ": ", str(e))

print("Addestramento terminato...")

while True:
    # cattura un singolo fotogramma
    ret, frame = video_cattura.read()
    
    # elabora solo un fotogramma video su due per risparmiare tempo
    if processa_questo_frame:
        # ridimensiona il fotogramma video a 1/4 di dimensione per un più rapido riconoscimento facciale
        piccolo_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # converti l'immagine dal colore BGR (che usa OpenCV) all'RGB (che usa face_recognition)
        rgb_piccolo_frame = piccolo_frame[:, :, ::-1]
        
        # trova tutti i volti e le codifiche dei volti nel fotogramma video corrente
        posizioni_volti = face_recognition.face_locations(rgb_piccolo_frame)
        codifiche_volti = face_recognition.face_encodings(rgb_piccolo_frame, posizioni_volti)
        
        nomi_volti = []
        for codifica_volto in codifiche_volti:
            # verifica se il volto corrisponde a uno dei volti conosciuti
            corrispondenze = face_recognition.compare_faces(codifiche_volti_conosciuti, codifica_volto)
            nome = "Sconosciuto"
            
            # se esistono volti conosciuti, allora procedi con il riconoscimento
            if len(codifiche_volti_conosciuti) > 0:
                # usa la minima distanza dal nuovo volto
                distanze_volti = face_recognition.face_distance(codifiche_volti_conosciuti, codifica_volto)
                indice_migliore_corrispondenza = np.argmin(distanze_volti)
                if corrispondenze[indice_migliore_corrispondenza]:
                    nome = nomi_volti_conosciuti[indice_migliore_corrispondenza]
            
            nomi_volti.append(nome)
    
    processa_questo_frame = not processa_questo_frame
    
    # mostra i risultati (lento)
    for (alto, destro, basso, sinistro), nome in zip(posizioni_volti, nomi_volti):
        # scala le posizioni dei volti dato che il fotogramma in cui abbiamo rilevato era scalato a 1/4 di dimensione
        alto *= 4
        destro *= 4
        basso *= 4
        sinistro *= 4
        
        # disegna un rettangolo attorno al volto
        cv2.rectangle(frame, (sinistro, alto), (destro, basso), (0, 0, 255), 2)
        
        # disegna un'etichetta con un nome sotto il volto
        cv2.rectangle(frame, (sinistro, basso - 35), (destro, basso), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, nome, (sinistro + 6, basso - 6), font, 1.0, (255, 255, 255), 1)
        
        if "Sconosciuto" == nome:
            cv2.putText(frame, "ALLARME!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            # aggiungere codice a piacere in caso di allarme attivato
    
    # mostra l'immagine risultante
    cv2.imshow('Video', frame)
    
    # premi 'q' sulla tastiera per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# rilascio della webcam
video_cattura.release()
cv2.destroyAllWindows()
