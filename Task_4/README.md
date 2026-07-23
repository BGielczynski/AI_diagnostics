# Task 4 - Ein-Klassen-Klassifikation

Die Aufgabe vergleicht zwei getrennte 4x2-Pipelines. Jede Pipeline trainiert
fuer die vier vorgegebenen Leave-P-Groups-Out-Splits je einen Autoencoder fuer
`Ch1` und `Ch2`.

- `per_measurement`: Jede WAV-Datei bleibt nach der Feature-Extraktion ein Sample.
- `mid_averaged`: Die extrahierten Features werden danach wie in Task 3 innerhalb
  von `spec`, `pos`, `rID` und Kanal ueber `mID` gemittelt.

Beide Varianten verwenden dieselben 62 Audiofeatures, Modellparameter, Splits
und Metriken. Dadurch ist der Einfluss der mID-Mittelung direkt vergleichbar.

## Ausfuehrung

Aus dem Repository-Stamm:

```powershell
python Task_4/src/main.py
```

Optional kann nur eine Pipeline ausgefuehrt werden:

```powershell
python Task_4/src/main.py --pipeline per_measurement
python Task_4/src/main.py --pipeline mid_averaged
```

Mit `--force-extract` werden die Audiofeatures neu extrahiert.

## Ergebnisstruktur

```text
Task_4/results/
|-- shared/             gemeinsamer Featurecache vor der mID-Mittelung
|-- per_measurement/    acht Modelle und Auswertungen ohne Mittelung
|-- mid_averaged/       acht Modelle und Auswertungen mit mID-Mittelung
`-- comparison/         direkter Vergleich beider Pipelines
```

Labelkonvention: Z01-Z04 sind gesund (`1`), Z05 ist anomal/beschaedigt (`0`).
Die Autoencoder werden ausschliesslich auf gesunden Samples trainiert. Die
Anomalieschwelle entspricht dem 98%-Quantil der jeweiligen Trainingsfehler.
