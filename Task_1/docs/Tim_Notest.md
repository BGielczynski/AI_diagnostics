Wahl der Fensterfunktion:
'hann': gute Ergebnisse, oft verschwommen aber unterschiede waren gut identifizierbar
'hamming': keine offensichtliche unterschiede zu hann festgestellt
'blackman': keine offensichtliche unterschiede zu hann festgestellt
'boxcar': sehr "dünne" signale, periodische veränderung aúnd abklingen des Signals sind nicht mehr gut erkennbar

=> im weiteren Verlauf größtenteil 'hann' genutzt

Wahl der Fensterlänge: Mit Fensterlänge von 400 messpunkten gestartet
- bei Verringerung der Fensterlänge (um 50 Einheiten; bis 200 getestet) wurden die spektogramme "gestreckter" und kleine unterschiede wurden schwerer zu erkennen.
- bei erhöhung wurden die Signale zwar genauer, aber auch kleiner im spekrogramm und somit auch schwehrer zu lesen.

=> Optimalwert bei ungefähr 500 - 600 => 550 finaler Wert

Überlappung: Bei 50% gestartet.
- niedrigere Überlappung (bis 0 getestet in 10er schritten) macht periodische änderungen in der häufigkeit der Frequenzen etwas schlechter sichtbar.
- bei zu hoher überlappung wird das spektogramm ebenfalls etwas verschwommener -> Details gehen verloren?

=> insgesamt sahen die spektrogramme bei einer Überlappung von 50% am besten aus, änderungen haben entweder den detailgrad verschlechtert oder zu Informationsverlust geführt.