# Praesentationsleitfaden Aufgabe 3

## Roter Faden

Wir zeigen, wie aus den Rohsignalen ein fairer Datensatz entsteht, wie daraus Merkmale extrahiert werden, wie das MLP trainiert und optimiert wird und wie die Ergebnisse kritisch zu bewerten sind.

Wichtig fuer die Code-Vorstellung: Alles, was aus Aufgabe 1 und 2 bekannt ist, nur kurz einordnen. Ausfuehrlicher zeigen wir den neuen Code aus Aufgabe 3: Aggregation, Split, MLP-Training, Hyperparameter-Suche, Evaluation und GIF-Erzeugung.

## Folie 1: Ziel der Aufgabe

**Bild:** keins

**Text auf der Folie:**

- Binaere Klassifikation von Zahnrad-Audiosignalen
- Modell: Multilayer Perceptron (MLP)
- Klassen:
    - `1 = gut`
    - `0 = beschaedigt`
- Bewertung mit Testsatz, ROC und AUC

**Notizen:**

In Aufgabe 3 geht es darum, die Daten aus Aufgabe 2 nicht nur visuell zu untersuchen, sondern automatisch zu klassifizieren. Das Ziel ist eine binaere Entscheidung: Ist ein Signal einem guten oder einem beschaedigten Zahnrad zuzuordnen? Dafuer verwenden wir ein MLP und bewerten das Modell erst am Ende auf einem separaten Testsatz.

## Folie 2: Datensatz und Label-Zuordnung

**Bild:** optional Tabelle aus Split-/Datenuebersicht

**Text auf der Folie:**

- Datenquelle: `Task_2/data/sig`
- Proben: `Z01` bis `Z05`
- Label-Zuordnung:
    - `Z01`, `Z02`, `Z03`, `Z04` = gut
    - `Z05` = beschaedigt
- Kanaele: `Ch1`, `Ch2`

**Notizen:**

Die Label-Zuordnung basiert auf den Ergebnissen aus Aufgabe 2. Dort wurde `Z05` als beschaedigt betrachtet, waehrend `Z01` bis `Z04` als gut verwendet werden. Wichtig ist, dass nur eine Probe die Klasse 0 bildet. Das ist fuer die spaetere Diskussion relevant, weil das Modell dadurch nicht viele verschiedene beschaedigte Zahnraeder sieht.

## Folie 3: Problem vor der Modellierung

**Bild:** keins oder kleine schematische Tabelle

**Text auf der Folie:**

- Unterschiedlich viele `mID` pro Zahnrad
- `Z01` und `Z04` haben mehr Wiederholungsmessungen
- Gefahr: Modell gewichtet diese Proben staerker
- Loesung: Mittelung ueber `mID`

**Notizen:**

Ein direktes Training auf allen Dateien waere problematisch. `Z01` und `Z04` besitzen mehr Wiederholungsmessungen als `Z02`, `Z03` und `Z05`. Wenn wir jede Datei als unabhaengigen Datenpunkt verwenden, bekommen diese Proben mehr Gewicht. Deshalb mitteln wir die Merkmale ueber `mID`. Bei Zahnraedern mit nur einer `mID` veraendert sich dadurch nichts.

## Folie 4: Pipeline vom Signal zum Datensatz

**Bild:** optional eigenes Pipeline-Schaubild oder Textdiagramm

**Text auf der Folie:**

```text
WAV-Dateien
-> DataFrame mit Metadaten
-> Merkmalsextraktion
-> mID-Mittelung
-> Channel-One-Hot
-> Train / Dev / Test
```

**Notizen:**

Die Verarbeitung ist in mehrere Module aufgeteilt. `dataframe_manager.py` laedt Signale und Metadaten. `extraction.py` berechnet die Merkmale. `aggregation.py` mittelt ueber die Messungs-ID. Danach wird der Channel als Zusatzinformation aufgenommen. Anschliessend erzeugt `split.py` die Trainings-, Entwicklungs- und Testdaten.

## Folie 5: Merkmalsextraktion

**Bild:** optional kein Bild, oder Boxplot/PCA aus Task 2 falls gewuenscht

**Text auf der Folie:**

- Gleiche Merkmalsfamilien wie in Aufgabe 2
- MFCC und Delta-MFCC
- Spektralzentrum, Bandbreite, Roll-off
- Spektralkontrast
- Chroma-STFT und Chroma-CENS
- Zero-Crossing-Rate und RMS-Energie

**Notizen:**

Wir verwenden bewusst die Merkmale aus Aufgabe 2 weiter, weil diese bereits zu guten visuellen Trennungen gefuehrt haben. Die Merkmale beschreiben sowohl zeitliche Eigenschaften als auch spektrale Eigenschaften des Signals. Dadurch muss das MLP nicht direkt auf den Rohsignalen lernen, sondern bekommt kompakte Signalbeschreibungen.

## Folie 6: mID-Mittelung und Channel-Behandlung

**Bild:** keins oder kleine Tabelle

**Text auf der Folie:**

- Mittelung ueber:
    - `mID`
- Getrennt bleiben:
    - `spec`
    - `pos`
    - `rID`
    - `sID`
- Channel als Feature:
    - `sID_Ch1`
    - `sID_Ch2`

**Notizen:**

Wir mitteln nicht die Rohsignale, sondern die extrahierten Merkmale. Dadurch entstehen stabilere Werte und eine ausgeglichenere Datenbasis. `Ch1` und `Ch2` bleiben getrennte Datenpunkte, weil ein einzelner Channel spaeter auch einzeln bewertet werden koennen soll. Damit das Modell trotzdem weiss, welcher Channel vorliegt, geben wir `sID` als One-Hot-Feature mit.

## Folie 7: Train-, Dev- und Testsplit

**Bild:** optional Tabelle aus Konsolenausgabe oder `features_split.csv`

**Text auf der Folie:**

- Split-Verhaeltnis: 60 / 20 / 20
- Gruppierter Split nach:

```text
spec + pos + rID
```

- Ch1 und Ch2 derselben Messung bleiben zusammen
- Ergebnis pro Zahnrad und Channel:
    - 12 Training
    - 4 Entwicklung
    - 4 Test

**Notizen:**

Der Split ist ein wichtiger methodischer Punkt. Wir splitten nicht zufaellig einzelne Dateien, weil sonst Ch1 und Ch2 derselben Messung in unterschiedlichen Mengen landen koennten. Das waere Datenleckage. Stattdessen bleiben zusammengehoerige Messsituationen im selben Split. Der Entwicklungssatz wird nur fuer die Hyperparameter-Auswahl verwendet, der Testsatz erst ganz am Ende.

## Folie 8: MLP-Modell

**Bild:** optional kein Bild

**Text auf der Folie:**

- Pipeline:

```text
StandardScaler -> MLPClassifier
```

- Skalierung nur auf Training fitten
- Finale Features: Signalmerkmale + Channel-One-Hot
- Kleine Netze wegen kleinem Datensatz

**Notizen:**

Vor dem MLP werden alle Features standardisiert. Das ist wichtig, weil die Merkmale unterschiedliche Wertebereiche besitzen. Das finale Modell ist bewusst klein gehalten, da der Datensatz nach der Aggregation nur 200 Datenpunkte enthaelt. Ein zu grosses Netz koennte leicht auswendig lernen.

## Folie 9: Hyperparameter-Suche

**Bild:** `results/top_models.csv` als Tabelle oder Screenshot

**Text auf der Folie:**

- Optimierung auf Entwicklungssatz
- Getestet:
    - Hidden Layer
    - Lernrate
    - Batchgroesse
    - Regularisierung `alpha`
- Bestes Modell:

```text
hidden_layer_sizes = (16,)
learning_rate_init = 0.0001
batch_size = 8
alpha = 0.0001
```

**Notizen:**

Die Hyperparameter werden nicht auf dem Testsatz ausgewaehlt. Fuer jede Kombination trainieren wir auf dem Trainingssatz und bewerten auf dem Entwicklungssatz. Das beste Modell wird anhand der Entwicklungsmetriken ausgewaehlt. Danach wird das finale Modell mit Training plus Entwicklung neu trainiert und erst dann auf dem Testsatz bewertet.

## Folie 10: Neuer Code in Aufgabe 3

**Bild:** optional Screenshot der Projektstruktur `Task_3/src`

**Text auf der Folie:**

- Neu in Task 3:
    - `aggregation.py`
    - `split.py`
    - `model.py`
    - `create_training_gif.py`
- Angepasst:
    - `main.py`
    - `dataframe_manager.py`
    - `extraction.py`

**Notizen:**

Bei der Code-Vorstellung sollten wir nicht lange den bekannten DataFrame-Aufbau aus Aufgabe 1 und 2 erklaeren. Neu und wichtig sind vor allem die mID-Mittelung in `aggregation.py`, der gruppierte Split in `split.py`, die komplette MLP-Logik in `model.py` und die GIF-Erzeugung in `create_training_gif.py`. `main.py` verbindet alle Schritte zu einer Pipeline.

## Folie 11: Code-Ausschnitt Aggregation und Split

**Bild:** optional Code-Screenshot aus `aggregation.py` und `split.py`

**Text auf der Folie:**

- `aggregation.py`
    - Mittelwert ueber `mID`
    - `spec`, `pos`, `rID`, `sID` bleiben getrennt
- `split.py`
    - gruppiert nach `spec + pos + rID`
    - verhindert Channel-Leakage

**Notizen:**

Hier zeigen wir konkret, warum die Datenbasis fairer wird. In `aggregation.py` werden die extrahierten Merkmale gemittelt, nicht die Rohsignale. In `split.py` wird verhindert, dass Ch1 und Ch2 derselben Messsituation getrennt in Training und Test landen. Das ist ein wichtiger fachlicher Punkt und sollte im Code kurz gezeigt werden.

## Folie 12: Code-Ausschnitt MLP und Evaluation

**Bild:** optional Code-Screenshot aus `model.py`

**Text auf der Folie:**

- `StandardScaler -> MLPClassifier`
- Hyperparameter-Suche auf Entwicklungssatz
- Finale Bewertung auf Testsatz
- Ausgabe:
    - Metriken
    - Confusion Matrix
    - ROC/AUC
    - Loss-Kurve

**Notizen:**

`model.py` ist der wichtigste neue Code fuer Aufgabe 3. Dort wird festgelegt, welche Spalten als Features ins Modell gehen, wie die Pipeline gebaut wird, wie Hyperparameter getestet werden und welche Ergebnisse gespeichert werden. Wichtig ist: Der Testsatz wird nicht zur Auswahl der Hyperparameter verwendet.

## Folie 13: Training pro Iteration

**Bild:** `results/training_iteration_demo.gif`

**Text auf der Folie:**

- Loss sinkt waehrend des Trainings
- Pro Iteration:
    - Vorhersage
    - Fehlerberechnung
    - Gewichtsupdate
- Training endet nach Konvergenz

**Notizen:**

Das GIF zeigt die echte Loss-Kurve des besten MLP. In jeder Iteration macht das Modell Vorhersagen, berechnet den Fehler und passt seine Gewichte per Backpropagation an. Der sinkende Loss zeigt, dass das Modell die Trainingsdaten zunehmend besser beschreibt.

## Folie 14: Training- und Entwicklungs-Accuracy

**Bild:** `results/accuracy_curve_train_dev.png`

**Text auf der Folie:**

- Training Accuracy und Dev Accuracy pro Iteration
- Beide Kurven steigen auf `1.00`
- Kein sichtbares Overfitting im aktuellen Split
- Dev-Satz dient zur Hyperparameter-Auswahl

**Notizen:**

Dieser Plot zeigt, ob das Modell nur die Trainingsdaten besser lernt oder auch auf dem Entwicklungssatz besser wird. Wenn Training steigt und Dev faellt, waere das ein Hinweis auf Overfitting. Hier steigen beide Kurven bis auf 1.0. Das spricht fuer eine klare Trennung im aktuellen Feature-Raum, muss aber wegen der kleinen Datenbasis weiter kritisch eingeordnet werden.

## Folie 15: Entscheidungsgrenze im Training

**Bild:** `results/training_decision_boundary.gif`

**Text auf der Folie:**

- Punkte: Trainingsdaten in 2D-PCA-Projektion
- Linie: Entscheidungsgrenze des MLP
- Rot: beschaedigt `0`
- Blau: gut `1`

**Notizen:**

Diese Darstellung ist eine Visualisierung. Die Punkte sind auf zwei PCA-Achsen reduziert, damit wir sie anzeigen koennen. Die Entscheidungsgrenze stammt aber vom MLP im vollstaendigen Merkmalsraum und wird in die PCA-Ebene projiziert. Deshalb ist die Grafik gut zur Erklaerung des Lernprozesses, aber nicht der eigentliche Trainingsraum.

## Folie 16: Testergebnisse

**Bild:** `results/confusion_matrix_test.png`

**Text auf der Folie:**

- Finale Bewertung auf Testsatz
- Aktuelle Testmetriken:
    - Accuracy: `1.00`
    - Balanced Accuracy: `1.00`
    - F1 fuer beschaedigt: `1.00`
- Keine Fehlklassifikation im aktuellen Split

**Notizen:**

Die Confusion Matrix zeigt, dass im aktuellen Testsatz alle Beispiele korrekt klassifiziert wurden. Besonders wichtig ist der Recall fuer Klasse 0, also fuer beschaedigte Signale. Dieser ist hier 1.0. Trotzdem muessen wir das Ergebnis kritisch einordnen, weil der Datensatz klein ist.

## Folie 17: ROC und AUC

**Bild:** `results/roc_curve_test.png`

**Text auf der Folie:**

- ROC fuer Klasse `0 = beschaedigt`
- AUC: `1.00`
- Kurve liegt ideal links oben

**Notizen:**

Die ROC-Kurve zeigt, wie gut das Modell zwischen beschaedigt und gut trennt, wenn man den Entscheidungsschwellwert veraendert. Eine AUC von 1.0 bedeutet perfekte Trennung im aktuellen Testsatz. Auch hier gilt: Das ist ein starkes Ergebnis, aber wegen der Datenlage vorsichtig zu interpretieren.

## Folie 18: Kritische Diskussion

**Bild:** keins

**Text auf der Folie:**

- Sehr gute Ergebnisse im aktuellen Split
- Aber:
    - kleiner Datensatz
    - nur `Z05` ist beschaedigt
    - Testdaten stammen aus denselben Proben
    - Generalisierung auf neue Zahnraeder offen

**Notizen:**

Der wichtigste Diskussionsteil ist die Einordnung. Das Modell kann die vorhandenen Daten sehr gut trennen. Es ist aber nicht bewiesen, dass es auf beliebige neue Zahnraeder generalisiert. Dafuer braeuchte man mehr beschaedigte Proben und idealerweise einen Test mit komplett unbekannten Zahnraedern.

## Folie 19: Fazit

**Bild:** optional keins

**Text auf der Folie:**

- Saubere Pipeline von Rohsignal zu MLP
- Faire Datenbasis durch mID-Mittelung
- Gruppierter Split gegen Datenleckage
- MLP erreicht perfekte Testwerte im aktuellen Datensatz
- Naechster Schritt: mehr beschaedigte Proben testen

**Notizen:**

Zusammenfassend haben wir eine vollstaendige Klassifikationspipeline aufgebaut. Die wichtigsten methodischen Entscheidungen waren die Aggregation ueber mID und der gruppierte Split. Das Ergebnis ist sehr gut, muss aber wegen der begrenzten Datenbasis kritisch diskutiert werden. Fuer eine robustere Aussage waeren weitere beschaedigte Zahnraeder notwendig.
