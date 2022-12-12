TO-DO:
  Confrontarsi con lo stato dell'arte algoritmi di feature extraction che lavorano bene nel task dell'iris recognition o meglio in
  generale sulle texture (e.g. Local Binary Pattern);
  Individuati gli algoritmi vedere se ci sono implementazioni, provando a buttare due righe di codice;
  Costruire modello CNN da allenare con le immagini normalizzate secche;
  In caso di feature extraction usare modello diverso da CNN da allenare;



considerazioni strade possibili:
a. implementare modulo di feature extraction (e.g. LBP) + modello di machine learning 
b. implementare CNN (feature extraction eseguito tramite convoluzioni) con possibilità di utilizzare 
   variante (e.g. LBCNN https://github.com/juefeix/lbcnn.pytorch)

    TO-DO caso a:
    Effettuare feature extraction di massa per ogni elemento del dataset e fare labelling
    Scegliere modello, fare training ed evaluation

    TO-DO caso b:
    Costruzione del dataset tramite torch Dataset e Dataloader con labelling tramite cartelle del dataset (nome Cartella = target)
    Costruzione del modello 


Nel caso di calcolo del modulo di feature extraction farei uno studio dello stato dell'arte per vedere quali algoritmi funzionano meglio per il nostro task.


