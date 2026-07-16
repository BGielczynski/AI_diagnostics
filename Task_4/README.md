# Task 4 – Ein-Klassen-Klassifikation

Die Aufgabe trainiert fuer jede der vier vorgegebenen Leave-P-Groups-Out-
Strategien je einen Autoencoder fuer `Ch1` und `Ch2` (acht Modelle insgesamt).
Jede WAV-Datei bleibt ein eigenes Sample; es erfolgt keine Mittelung ueber `mID`.

Ausfuehrung aus dem Repository-Stamm:

```powershell
python Task_4/src/main.py
```

Die erste Ausfuehrung extrahiert 62 Audiofeatures und speichert sie unter
`Task_4/results/features_per_measurement.csv`. Mit `--force-extract` wird dieser
Cache neu aufgebaut. Alle Einzel- und Aggregatmetriken, Vorhersagen, Modelle und
Abbildungen werden unter `Task_4/results` abgelegt.

Labelkonvention: Z01–Z04 sind gesund (`1`), Z05 ist anomal/beschaedigt (`0`).
Der Autoencoder wird ausschliesslich auf gesunden Samples trainiert. Ein Sample
gilt als anomal, wenn sein standardisierter Rekonstruktionsfehler groesser als
das 95%-Quantil der Trainingsfehler ist.
