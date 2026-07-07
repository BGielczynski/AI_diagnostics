# Praesentationsleitfaden Aufgabe 3

## Roter Faden

Wir zeigen, wie aus den Rohsignalen ein fairer Datensatz entsteht, wie daraus Merkmale extrahiert werden, wie das MLP trainiert und optimiert wird und wie die Ergebnisse kritisch zu bewerten sind.

Wichtig fuer die Code-Vorstellung: Alles, was aus Aufgabe 1 und 2 bekannt ist, nur kurz einordnen. Ausfuehrlicher zeigen wir den neuen Code aus Aufgabe 3: Feature-Mittelung ueber `mID`, zahnradbasierter Split mit separater Z05-Verteilung, zwei getrennte Channel-Modelle, Hyperparameter-Suche, Evaluation, Learning Curves und GIF-Erzeugung.

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

Ein direktes Training auf allen Dateien waere problematisch. `Z01` und `Z04` besitzen mehr Wiederholungsmessungen als `Z02`, `Z03` und `Z05`. Wenn wir jede Datei als unabhaengigen Datenpunkt verwenden, bekommen diese Proben mehr Gewicht. Deshalb extrahieren wir zuerst die Merkmale pro Datei und mitteln danach die Featurewerte ueber alle `mID` derselben Messsituation.

## Folie 4: Pipeline vom Signal zum Datensatz

**Bild:** optional eigenes Pipeline-Schaubild oder Textdiagramm

**Text auf der Folie:**

```text
WAV-Dateien
-> DataFrame mit Metadaten
-> zahnradbasierter Split
-> Merkmalsextraktion
-> Feature-Mittelung ueber mID
-> Ch1-/Ch2-Datensatz
-> je ein MLP pro Channel
-> Test-Routing nach Channel
```

**Notizen:**

Die Verarbeitung ist in mehrere Module aufgeteilt. `dataframe_manager.py` laedt Signale und Metadaten. Danach ordnet `split.py` jede Rohdatei anhand der Zahnrad-ID und bei `Z05` anhand fester Positionen Train, Entwicklung oder Test zu. `extraction.py` berechnet die Merkmale pro Datei. In `main.py` werden diese Merkmale anschliessend ueber alle `mID` derselben Messsituation gemittelt.

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

## Folie 6: Split und Channel-Behandlung

**Bild:** keins oder kleine Tabelle

**Text auf der Folie:**

- Feature-Mittelung ueber `mID`
- Merkmalsextraktion pro Datei vor der Mittelung
- Fester Split ueber Zahnraeder:
    - Train: `Z01`, `Z02`
    - Dev: `Z04`
    - Test: `Z03`
- `Z05` positionsbasiert auf alle Splits verteilt
- Zwei Modelle:
    - `Ch1`
    - `Ch2`

**Notizen:**

Wir extrahieren die Merkmale pro Rohdatei und mitteln danach die Featurewerte ueber alle `mID`. Dadurch werden Wiederholungsmessungen in der Merkmalsebene zusammengefasst. Der Split trennt die guten Zahnraeder staerker nach Zahnrad-Identitaet: `Z04` ist nur im Entwicklungssatz, `Z03` nur im Testsatz. `Z05` muss als einziges beschaedigtes Zahnrad weiterhin positionsbasiert auf Train, Dev und Test verteilt werden. `Ch1` und `Ch2` werden nicht als Feature codiert, sondern bekommen je ein eigenes Modell.

## Folie 7: Train-, Dev- und Testsplit

**Bild:** `docs/Tsk4_DistrDiagramm.png`

**Text auf der Folie:**

- Split nach Zahnrad-Identitaet:

```text
Train: Z01, Z02 + Z05-Positionen Pos00, Pos01, Pos02, Pos04, Pos06, Pos08
Dev:   Z04      + Z05-Positionen Pos03, Pos05
Test:  Z03      + Z05-Positionen Pos07, Pos09
```

- Das Diagramm zeigt die Verteilung nach Channel:
    - rot = `Ch1`
    - blau = `Ch2`
- `Z01` und `Z04` werden wegen vieler Wiederholungen gemittelt:
    - `Z01`: 100 Rohsignale pro Channel -> 20 Feature-Zeilen
    - `Z04`: 60 Rohsignale pro Channel -> 20 Feature-Zeilen
- `Z02`, `Z03` und `Z05` haben bereits 20 Rohsignale pro Channel
- Ergebnis nach `mID`-Mittelung pro Channel:
    - Train: 52
    - Dev: 24
    - Test: 24

**Notizen:**

Das Verteilungsdiagramm macht sichtbar, warum wir vor dem Training mitteln. `Z01` und `Z04` haetten sonst deutlich mehr Einfluss als die anderen Zahnraeder, weil dort mehr Wiederholungsmessungen vorhanden sind. Nach der Mittelung stehen fuer jedes gute Zahnrad pro Channel 20 Feature-Zeilen zur Verfuegung. Danach werden die guten Zahnraeder getrennt: `Z01` und `Z02` liegen im Training, `Z04` im Entwicklungssatz und `Z03` im Testsatz. Fuer die beschaedigte Klasse ist eine solche Zahnrad-Trennung nicht moeglich, weil nur `Z05` beschaedigt ist. Deshalb wird `Z05` pro Channel positionsbasiert in 12 Training, 4 Entwicklung und 4 Test aufgeteilt. Daraus entstehen pro Channel 52 Trainings-, 24 Entwicklungs- und 24 Testbeispiele.

## Folie 8: MLP-Modell

**Bild:** optional kein Bild

**Text auf der Folie:**

- Pipeline je Channel:

```text
StandardScaler -> MLPClassifier
```

- Skalierung nur auf Training fitten
- Finale Features: nur Signalmerkmale des jeweiligen Channels
- Kleine Netze wegen kleinem Datensatz

**Notizen:**

Vor dem MLP werden alle Features standardisiert. Das ist wichtig, weil die Merkmale unterschiedliche Wertebereiche besitzen. Pro Channel trainieren wir ein eigenes kleines Modell, da der Datensatz trotz Rohdateien klein bleibt. Ein zu grosses Netz koennte leicht auswendig lernen.

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

## Folie 10: Code-Ueberblick

**Bild:** keins

**Text auf der Folie:**

- `main.py`: steuert die komplette Pipeline
- `dataframe_manager.py`: laedt WAV-Dateien und Metadaten
- `split.py`: ordnet Rohsignale Train, Dev und Test zu
- `extraction.py`: berechnet Signalmerkmale pro Datei
- `model.py`: trainiert, optimiert und bewertet die MLP-Modelle

**Notizen:**

Die Code-Vorstellung laeuft am besten ueber `main.py`, weil dort alle Schritte sichtbar zusammenlaufen: Signale laden, Split anwenden, Features extrahieren, ueber `mID` mitteln, pro Channel ein Modell trainieren und am Ende Ergebnisse speichern. Die anderen Dateien erklaeren wir kurz nach ihrer Rolle. `dataframe_manager.py` baut den Rohdaten-DataFrame aus den WAV-Dateien. `split.py` enthaelt die methodisch wichtige Aufteilung: `Z01` und `Z02` ins Training, `Z04` in den Entwicklungssatz, `Z03` in den Testsatz und `Z05` positionsbasiert in alle drei Splits. `extraction.py` berechnet die Merkmale wie MFCC, Spektralmerkmale, Chroma, Zero-Crossing-Rate und RMS. `model.py` muss nicht im Detail vorgestellt werden; wichtig ist nur, dass dort `StandardScaler -> MLPClassifier`, Hyperparameter-Suche, finale Evaluation, ROC, Confusion Matrix, Learning Curves und Loss-Kurven umgesetzt sind.

## Folie 11: Learning Curves

**Bild:** `results/model_Ch1/learning_curve.png` und `results/model_Ch2/learning_curve.png`

**Text auf der Folie:**

- Train- und Dev-Score pro Trainingsmenge
- Kurven zeigen Datenbedarf des Modells
- Bereits kleine Trainingsmenge trennt fast perfekt
- Ch1 erreicht frueh perfekte Dev-Werte
- Ch2 braucht im aktuellen Split die volle Trainingsmenge fuer perfekte Dev-Werte

**Notizen:**

Die Learning Curves zeigen, wie gut das Modell mit wachsender Trainingsmenge wird. Im aktuellen Split erreicht `Ch1` schon mit kleinen Trainingsmengen perfekte Entwicklungswerte. `Ch2` verbessert sich schrittweise und erreicht die perfekten Entwicklungswerte erst mit der gesamten Trainingsmenge.

## Folie 12: Metriken je Channel

**Bild:** Tabellen aus `results/model_Ch1/final_test_metrics.csv` und `results/model_Ch2/final_test_metrics.csv`

**Text auf der Folie:**

- Ch1 und Ch2 separat bewertet
- Accuracy, Balanced Accuracy, F1
- Specificity, NPV, MCC
- Beide Channel erreichen aktuell `1.00`

**Notizen:**

Die Channel werden separat trainiert und bewertet. Neben Accuracy verwenden wir auch robustere Kennzahlen wie Balanced Accuracy und MCC. Die perfekten Werte muessen dennoch vorsichtig interpretiert werden, weil nur `Z05` die beschaedigte Klasse bildet.

## Folie 13: Entscheidungsgrenze im Training

**Bild:** `results/model_Ch1/training_decision_boundary.gif` und `results/model_Ch2/training_decision_boundary.gif`

**Text auf der Folie:**

- Punkte: Trainingsdaten in 2D-PCA-Projektion
- Linie: lineare Entscheidungsgrenze in der PCA-Ebene
- Rot: beschaedigt `0`
- Blau: gut `1`

**Notizen:**

Diese Darstellung ist eine Visualisierung. Die Punkte sind auf zwei PCA-Achsen reduziert, damit wir sie anzeigen koennen. Die eingezeichnete Trennlinie ist bewusst linear und zeigt die Trennung in dieser 2D-Projektion. Das eigentliche MLP arbeitet weiterhin im vollstaendigen Merkmalsraum.

## Folie 14: Testergebnisse

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

## Folie 15: ROC und AUC

**Bild:** `results/roc_curve_test.png`

**Text auf der Folie:**

- ROC fuer Klasse `0 = beschaedigt`
- AUC: `1.00`
- Kurve liegt ideal links oben

**Notizen:**

Die ROC-Kurve zeigt, wie gut das Modell zwischen beschaedigt und gut trennt, wenn man den Entscheidungsschwellwert veraendert. Eine AUC von 1.0 bedeutet perfekte Trennung im aktuellen Testsatz. Auch hier gilt: Das ist ein starkes Ergebnis, aber wegen der Datenlage vorsichtig zu interpretieren.

## Folie 16: Kritische Diskussion

**Bild:** keins

**Text auf der Folie:**

- Sehr gute Ergebnisse im aktuellen Split
- Aber:
    - kleiner Datensatz
    - nur `Z05` ist beschaedigt
    - gute Testdaten stammen aus unbekanntem Zahnrad `Z03`
    - beschaedigte Klasse stammt weiterhin nur aus `Z05`
    - Generalisierung auf neue beschaedigte Zahnraeder offen

**Notizen:**

Der wichtigste Diskussionsteil ist die Einordnung. Der neue Split prueft die guten Zahnraeder strenger, weil `Z03` im Test nicht im Training vorkommt. Fuer die beschaedigte Klasse bleibt die Aussage begrenzt, weil alle beschaedigten Signale aus `Z05` stammen. Fuer eine robuste Aussage braeuchte man weitere beschaedigte Zahnraeder.

## Folie 17: Fazit

**Bild:** optional keins

**Text auf der Folie:**

- Saubere Pipeline von gemittelten Merkmalen zu zwei Channel-MLPs
- Zahnradbasierter Split statt zufaelligem Dateisplit
- Testsignale werden nach Channel geroutet
- Beide MLPs erreichen perfekte Testwerte im aktuellen Datensatz
- Naechster Schritt: mehr beschaedigte Proben testen

**Notizen:**

Zusammenfassend haben wir eine vollstaendige Klassifikationspipeline aufgebaut. Die wichtigsten methodischen Entscheidungen waren die Feature-Mittelung ueber `mID`, der zahnradbasierte Split und zwei getrennte Modelle fuer `Ch1` und `Ch2`. Das Ergebnis ist sehr gut, muss aber wegen der begrenzten Datenbasis kritisch diskutiert werden. Fuer eine robustere Aussage waeren weitere beschaedigte Zahnraeder notwendig.

