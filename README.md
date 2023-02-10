# Iris_Recognition_BS

Progetto per il corso "Biometric Systems" presso Sapienza Università di Roma

Contenuto delle cartelle:

• Dataset:
    • CASIA1:                   108 cartelle ciascuna contenente 7 diverse foto dello stesso occhio, catturate in 2 sessioni
    • CASIA_Iris_interval_norm: contiene i risultati delle immagini normalizzate organizzate nelle corrispondenti cartelle del database di partenza CASIA1

• Pre-Processing:
    • segmentation.py:          segmenta l'immagine dell'occhio, andando a cerchiare l'iride e la pupilla
    • normalization.py:          divide l'immagine dell'occhio in rettangoli e salva le immagini frammentate nella dir CASIA_Iris_interval_norm

• KNN Model:
    • implementazione del modello KNN 

• CNN and VGG Model:
    • implementazione del modello CNN
    • implementazione del modello VGG16

• Demo:
    • implementazione di una demo riguardante il progetto utilizzando il modello CNN
