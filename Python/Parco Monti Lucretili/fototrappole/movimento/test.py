# Coded by Pietro Squilla
# Lo script va avviato in una cartella di soli video
import cv2
import os
import numpy
import shutil
import statistics

print("******************************************")
print("*                                        *")
print("*            Servizio Civile             *")
print("*     Progetto Monitoraggi faunisti      *")
print("*    Monti Lucretili, anno 2021/2022     *")
print("*                                        *")
print("******************************************\n")

# creo tante dir quanti sono i video (per analizzare i frame)
dirCorrente = os.getcwd()
listaFile = os.listdir()
listaFile.remove(os.path.basename(__file__))    # elimina il nome dello script dalla lista dei file video
numeroFile = (len(listaFile))

print("Creazione directory...")
for i in range(0,numeroFile):
    dirNuova = ".video_%s" %(listaFile[i])
    os.mkdir(dirNuova)

# cambio dir e analizzo i video uno dopo l'altro
for i in range(0,numeroFile):
    try:
        print("Estrapolazione frame video '%s'" %(listaFile[i]))
        vidcap = cv2.VideoCapture(listaFile[i])
        count = 1
        success = True
        os.chdir(".video_%s" %(listaFile[i]))
        
        try:
            while success:
                success,image = vidcap.read()
                cv2.imwrite("frame%d.jpg" %count,image)    # salva il frame in jpg
                if cv2.waitKey(10) == 27:                  # esci se è premuto Escape
                    break
                count += 1
        except:
            # devo tornare alla dir precedente
            os.chdir("..")
    except:
        continue

# ANALISI FRAME #
print("Analisi frame...")

# i -> file
# j -> frame
for i in range(0,numeroFile):
    print("\nVIDEO %s\nElaborazione pixel dei frame..." %(listaFile[i]))
    os.chdir(".video_%s" %(listaFile[i]))
    
    listaFrame = os.listdir()
    numeroFrame = len(listaFrame)
    
    # risoluzione frame
    x = 1920
    y = 1080
    sogliaLista = []
    flag1 = 0
    flag2 = 0
    
    # confronto i tutti i frame con il primo
    for j in range(1,numeroFrame):
        # il primo frame fa da background
        if j == 1:
            bg = cv2.imread(listaFrame[0])
            # scala grigi + blur + crop
            bg = cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)
            #bg = cv2.GaussianBlur(bg,(5,5),0)
            bg = cv2.blur(bg,(5,5))
            bg = bg[0:(y-60),0:x]
        
        # i frame successivi vengono comparati con il primo
        img = cv2.imread(listaFrame[j])
        # scala grigi + blur + crop
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img = cv2.GaussianBlur(img,(5,5),0)
        img = cv2.blur(img,(5,5))
        img = img[0:(y-60),0:x]
        
        # setta una soglia sul frame differenza
        delta = cv2.absdiff(bg,img)
        #threshold = cv2.threshold(delta,25,255,cv2.THRESH_BINARY)
        
        soglia = numpy.sum(delta)
        
        print(soglia)
        
        if soglia >= 11000000:
            flag1 = 1
            print("MOVIMENTO RILEVATO!")
            break
    
    # se non è stato rilevato nessun movimento faccio lo stesso confronto prendendo l'ultimo frame come background
    if flag1 == 0:
        # confronto tutti i frame con l'ultimo se non è
        for j in range(1,numeroFrame):
            # l'ultimo frame fa da background
            if j == 1:
                bg = cv2.imread(listaFrame[numeroFrame-1])
                # scala grigi + blur + crop
                bg = cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)
                #bg = cv2.GaussianBlur(bg,(5,5),0)
                bg = cv2.blur(bg,(5,5))
                bg = bg[0:(y-60),0:x]
            
            # i frame precedenti vengono comparati con il primo
            img = cv2.imread(listaFrame[j-1])
            # scala grigi + blur + crop
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #img = cv2.GaussianBlur(img,(5,5),0)
            img = cv2.blur(img,(5,5))
            img = img[0:(y-60),0:x]
            
            # setta una soglia sul frame differenza
            delta = cv2.absdiff(bg,img)
            #threshold = cv2.threshold(delta,25,255,cv2.THRESH_BINARY)
            
            soglia = numpy.sum(delta)
            
            #print(soglia)
            
            if soglia >= 11000000:
                flag2 = 1
                print("MOVIMENTO RILEVATO!")
                break
        
    # esco dalla dir corrente perchè ho analizzato il video i-esimo
    os.chdir("..")

# salvare i video con movimento dentro ad una cartella apposita

# eliminare ricorsivamente tutte le dir usate per l'analisi
for i in range(0,numeroFile):
    shutil.rmtree(".video_%s" %(listaFile[i]))

print("\nFine.")
input("Premi INVIO per uscire...")
