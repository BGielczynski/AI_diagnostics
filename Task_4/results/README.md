# Ergebnisstruktur

- `shared/`: gemeinsamer Cache der 440 Featurevektoren vor einer Mittelung.
- `per_measurement/`: acht Autoencoder ohne mID-Mittelung.
- `mid_averaged/`: 200 gemittelte Featurevektoren und acht zugehoerige Autoencoder.
- `comparison/`: ausschliesslich direkte Vergleichstabellen und -grafiken.

Jeder Pipelineordner enthaelt dieselben Zusammenfassungen sowie je Fold und
Kanal einen eigenen Modellordner. Dadurch werden Modelle, Vorhersagen und
Abbildungen der beiden Varianten nicht miteinander vermischt.
