# Iris_Recognition_BS

Progetto per il corso "Biometric Systems" presso Sapienza Università di Roma

Contenuto delle cartelle:

• Dataset:

    • CASIA1:                   108 cartelle ciascuna contenente 7 diverse foto dello stesso occhio, catturate in 2 sessioni

• Face Localization

    • brute_force_match.py:     contiene un metodo che utilizza il brute force match di opencv per calcolare quanto due occhi siano simili tra loro
    • face_localization.py:     localizza la faccia e gli occhi tramite la fotocamera del pc
    • iris_localization.py:     localizza e presenta a video i due occhi zoomati, localizzati tramite fotocamera del pc
    • iris_match.py:            utilizza un differente modo di calcolare la similitudine tra due immagini

• Segmentation

    • segmentation.py:          segmenta l'immagine dell'occhio, andando a cerchiare l'iride e la pupilla
    • segmentation_match.py:    utilizza gli stessi metodi della classe precedente per segmentare 2 occhi e successivamente confrontarli

• Normalization

    • normalization.py:          divide l'immagine dell'occhio in rettangoli e salva le immagini frammentate nella dir CASIA_Iris_interval_norm

• CASIA_Iris_interval_norm:
    
    • contiene i risultati delle immagini normalizzate organizzate nelle corrispondenti cartelle del database di partenza CASIA1


Tutte le classi sono al momento indipendenti ed eseguibili in modo a se stante
