• Unire segmentation e normalization 
• Completare la parte di pre-processing
• Cominciare a vedersi quali modelli di machine-learning utilizzare (capire se usare modello complesso con Pytorch o modello più    
  semplice tramite Scikit-learn)


strade possibili:
a. implementare modulo di feature extraction (e.g. LBP) + modello di machine learning 
b. implementare CNN (feature extraction eseguito tramite convoluzioni) con possibilità di utilizzare variante (e.g. LB)
c. modulo feature extraction + metrica di distanza/similarità

    TO-DO caso a:
    Effettuare feature extraction di massa per ogni elemento del dataset e fare labelling
    Scegliere modello, fare training ed evaluation

    TO-DO caso b:
    Costruzione del dataset tramite torch Dataset e Dataloader con labelling tramite cartelle del dataset (nome Cartella = target)
    Costruzione del modello 

    TO-DO caso c:
    Effettuare feature extraction di massa per ogni elemento del dataset e fare labelling
    Calcolare metriche e fare evaluation

Nel caso di calcolo del modulo di feature extraction farei uno studio dello stato dell'arte per vedere quali algoritmi funzionano meglio per il nostro task.