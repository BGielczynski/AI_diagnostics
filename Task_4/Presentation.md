---
marp: true
paginate: true
title: "Task 4 - Anomalieerkennung mit Autoencodern"
---

# Task 4

## Ein-Klassen-Klassifikation von Zahnrad-Audiosignalen

**Zwei Vorverarbeitungsvarianten x vier Splits x zwei Kanaele = 16 Autoencoder**

Kernaussage: Z05 wird in beiden Pipelines vollstaendig erkannt. Die mID-Mittelung
hilft einzelnen Folds, verschlechtert aber andere gesunde Zielgruppen.

<!--
Notizen - 35 Sekunden:
Wir vergleichen nicht zwei unterschiedlich optimierte Modelle, sondern zwei
Vorverarbeitungen unter identischen Bedingungen. Jede Variante umfasst die acht
von der Aufgabe geforderten Kanal- und Foldmodelle.
-->

---

# Daten und zwei Pipelines

- Z01-Z04: **gesund (Label 1)**; Z05: **anomal (Label 0)**
- 440 WAV-Dateien, 62 Audiofeatures je Datei
- Ch1 und Ch2 werden getrennt modelliert

| Pipeline | Verarbeitung nach Feature-Extraktion | Samples |
|---|---|---:|
| Einzelmessung | keine Mittelung | 440 |
| mID-gemittelt | Mittel je `spec`, `pos`, `rID`, Kanal | 200 |

Nach Mittelung besitzt jede Z-Gruppe genau 20 Samples je Kanal. Z01 fasst je
Sample 5 mIDs, Z04 3 mIDs zusammen; Z02, Z03 und Z05 besitzen nur eine mID.

<!--
Notizen - 60 Sekunden:
Die Feature-Extraktion wird immer vor der Mittelung ausgefuehrt. Dadurch mitteln
wir Merkmalsvektoren und nicht Rohsignale. Die Mittelung veraendert nur Z01 und
Z04; die anderen Gruppen bleiben inhaltlich unveraendert.
-->

---

# Vorgegebene Kreuzvalidierung

| Fold | Gesundes Training | Test: gesund | Test: anomal |
|---|---|---|---|
| 1 | Z01, Z02, Z03 | Z04 | Z05 |
| 2 | Z01, Z02, Z04 | Z03 | Z05 |
| 3 | Z01, Z03, Z04 | Z02 | Z05 |
| 4 | Z02, Z03, Z04 | Z01 | Z05 |

Pro Pipeline und Fold werden zwei Modelle trainiert: Ch1 und Ch2. Z05 und die
jeweils ausgelassene gesunde Gruppe beeinflussen weder Skalierung noch Training
oder Schwellenwert.

<!--
Notizen - 55 Sekunden:
Die aeusseren Splits entsprechen exakt der Aufgabenstellung. Jeder gesunde
Zustand wird einmal als unbekannte gesunde Gruppe getestet. Z05 wird in jedem
Fold erneut ausgewertet.
-->

---

# Identisches Modell fuer einen fairen Vergleich

```text
WAV -> 62 Features -> optional mID-Mittel -> StandardScaler
    -> 32 -> 8 -> 32 -> 62 -> Rekonstruktionsfehler -> Schwelle
```

- dichter Autoencoder (`MLPRegressor`), ReLU und Adam
- Lernrate 0,001; L2-Regularisierung 0,0001
- maximal 2.000 Epochen, Early Stopping mit 15 %
- feste Architektur und Seed 42
- Training ausschliesslich mit Label 1

<!--
Notizen - 60 Sekunden:
Alle Hyperparameter bleiben zwischen den Pipelines gleich. So kann eine
Ergebnisveraenderung der mID-Mittelung zugeschrieben werden und nicht einer
anderen Modellwahl. Die Skalierung wird fuer jedes Modell neu aus seinem
gesunden Trainingsfold gelernt.
-->

---

# Entscheidung und Metriken

**Anomaliescore:** mittlerer quadratischer Rekonstruktionsfehler im
standardisierten Merkmalsraum

**Schwelle:** 95%-Quantil der Fehler im gesunden Training

- oberhalb der Schwelle: anomal (Label 0)
- keine Verwendung von Z05 zur Schwellenoptimierung
- Sensitivitaet fuer Z05 und Spezifitaet fuer gesunde Testgruppen
- Accuracy, Balanced Accuracy (BAR), F1 und ROC-AUC

**Primaerer Vergleich:** Makromittel der BAR ueber alle acht Modelle, weil die
ungefilterte Pipeline ungleiche Testgruppengroessen besitzt.

<!--
Notizen - 55 Sekunden:
Ein gepoolter Vergleich allein waere unfair: Ohne Mittelung hat Fold 4 wegen
Z01 besonders viele Testzeilen. BAR gleicht Klassen aus; das Makromittel gibt
zusaetzlich jedem der acht Modelle dasselbe Gewicht.
-->

---

# Pipeline A: einzelne Messzeilen

![Modellmetriken Einzelmessungen](results/per_measurement/model_metric_summary.png)

| Fold, Ch1 / Ch2 | Sensitivitaet | Spezifitaet | BAR |
|---|---:|---:|---:|
| 1 | 1,000 | 0,717 / 0,733 | 0,858 / 0,867 |
| 2 | 1,000 | 0,750 / 0,750 | 0,875 / 0,875 |
| 3 | 1,000 | 0,850 / 0,750 | 0,925 / 0,875 |
| 4 | 1,000 | 0,190 / 0,280 | 0,595 / 0,640 |

<!--
Notizen - 70 Sekunden:
Alle Z05-Samples werden gefunden. Fold 1 bis 3 sind gut, Fold 4 scheitert an
vielen gesunden Z01-Fehlalarmen. Ohne Mittelung bleibt die Variation der fuenf
Z01-mIDs sichtbar.
-->

---

# Direkter Vergleich beider Pipelines

![Balanced Accuracy beider Pipelines](results/comparison/balanced_accuracy_by_fold.png)

| BAR, Kanaele gepoolt | Einzelmessung | mID-gemittelt |
|---|---:|---:|
| Fold 1 | 0,863 | **1,000** |
| Fold 2 | **0,875** | 0,838 |
| Fold 3 | **0,900** | 0,538 |
| Fold 4 | 0,617 | **0,750** |
| Makromittel 8 Modelle | **0,814** | 0,781 |

Sensitivitaet in beiden Pipelines und allen Modellen: **1,000**.

<!--
Notizen - 85 Sekunden:
Die Mittelung ist kein genereller Gewinn. Sie loest Fold 1 komplett und verbessert
Fold 4 deutlich. Gleichzeitig wird Fold 3 stark schlechter. Im fairen
Modellmakro bleibt die Einzelmessung mit 0,814 gegenueber 0,781 vorne.
-->

---

# Warum kippt das Ergebnis nach der Mittelung?

![Fehlerverteilung mID-gemittelt, Fold 3 Ch1](results/mid_averaged/fold_3_Ch1/reconstruction_errors.png)

- Fold 4 profitiert: mID-Variation von Z01 wird geglaettet
- Fold 1 profitiert ebenfalls von geglaettetem Z04 im Test
- Fold 3 verschlechtert sich: Z02 bleibt ungemittelt, waehrend Z01 und Z04 im
  Training geglaettet werden
- Fold 3 Spezifitaet: Ch1 **0,050**, Ch2 **0,100**
- Die Mittelung veraendert damit die Definition des gelernten Normalzustands

<!--
Notizen - 80 Sekunden:
Die gruene Verteilung ist gesundes Z02, liegt aber fast komplett rechts von der
Schwelle. Das Modell kennt in diesem Fold teilweise geglaettete Normaldaten.
Ungemitteltes Z02 wirkt relativ dazu neu. Der Autoencoder erkennt erneut
Verteilungsabweichung, nicht direkt die Schadensursache.
-->

---

# Diskussion

- **Robust:** Beide Varianten erkennen 160/160 Z05-Testentscheidungen
- **Einzelmessung:** bessere Makro-BAR und stabiler in Fold 2/3
- **mID-Mittelung:** weniger Z01/Z04-Variation, bessere Folds 1/4 und hoehere
  gepoolte AUC von 0,970
- **Risiko der Mittelung:** inkonsistente Vorverarbeitung, weil nur Z01 und Z04
  mehrere mIDs besitzen
- **Grenze:** Die 95%-Trainingsschwelle generalisiert nicht immer auf eine neue
  gesunde Z-Gruppe

Naechster Schritt: gruppenbasierte Schwellenkalibrierung und Vergleich mit einer
One-Class SVM; beide Pipelines sollten als Ablationsvergleich erhalten bleiben.

<!--
Notizen - 60 Sekunden:
Das Ergebnis spricht nicht dafuer, Mittelung grundsaetzlich zu verbieten. Es
zeigt aber, dass die Zahl der mIDs mit der Z-Gruppe gekoppelt ist. Dadurch kann
die Mittelung einen Fold verbessern und einen anderen verschlechtern.
-->

---

# Fazit

1. Zwei sauber getrennte 4x2-Pipelines ergeben insgesamt 16 Autoencoder.
2. Z05 wird immer erkannt: **Sensitivitaet 1,000**.
3. Die mID-Mittelung verbessert Fold 1 und 4, verschlechtert Fold 2 und besonders Fold 3.
4. Im fairen Makromittel ist die Einzelmessung besser: **BAR 0,814 vs. 0,781**.

## Kernaussage

**mID-Mittelung reduziert Variation, kann aber selbst einen Domain Shift erzeugen,
wenn nicht alle Z-Gruppen ueber dieselbe Anzahl an mIDs verfuegen.**

<!--
Notizen - 40 Sekunden:
Fuer die Hauptaussage wuerden wir die Einzelmessung bevorzugen und die gemittelte
Pipeline als wichtigen Ablationsvergleich zeigen. Gesamte Sprechzeit: etwa zehn Minuten.
-->
