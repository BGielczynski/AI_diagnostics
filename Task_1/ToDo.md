[Noch zu tun]
- die Signale simmultan nebeneinander plotten mit beschriftung der einzelnen Signale
- die visualizer.py anpassen um dies zu ermöglichen, demnach aus der main rausnehmen
- Höhe der Dargestellten Frequenzen anpassen, damit der Signalunterschied genauer sichtbar ist
- Frequenzen normieren um dies zu ermöglichen, ggf. einen Offset im Spektrogramm anpassen 
- Window, nperseg und noverlap anpassen um die Dastellung zu optimieren
    - Was für eine Window ist am sinnvollsten?
    - Je kleiner das nperseg, desto höher die Auflösung in der Zeit, aber desto geringer die Auflösung in der Frequenz
    - Je größer das nperseg, desto geringer die Auflösung in der Zeit, aber desto höher die Auflösung in der Frequenz
    - Je kleiner der noverlap, desto geringer die Auflösung in der Zeit, aber desto höher die Auflösung in der Frequenz
    - Je größer der noverlap, desto höher die Auflösung in der Zeit, aber desto geringer die Auflösung in der Frequenz
    - Je mehr Samples, desto höher die Auflösung in der Zeit
    - Je weniger Samples, desto geringer die Auflösung in der Zeit
    