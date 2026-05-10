# Team-Workflow & Git-Struktur

Da ihr zu dritt an diesem Projekt arbeitet, ist es wichtig, klare Regeln für die Zusammenarbeit festzulegen. Diese Anleitung hilft euch dabei, Konflikte zu vermeiden und effizient Code zu schreiben.

## 1. Branching-Strategie (Feature Branch Workflow)
Arbeitet **niemals direkt auf dem `main`-Branch**. Der `main`-Branch enthält immer nur funktionierenden, stabilen Code.

Wenn du an einer neuen Aufgabe beginnst:
1. Wechsle auf den neuesten Stand von `main`:
   ```bash
   git checkout main
   git pull origin main
   ```
2. Erstelle einen neuen Branch für deine Aufgabe:
   ```bash
   git checkout -b feature/name-der-aufgabe
   # z.B. git checkout -b feature/task-1-data-preprocessing
   ```

**Namenskonventionen für Branches:**
- `feature/...` für neue Funktionen oder Aufgaben
- `bugfix/...` für Fehlerbehebungen
- `docs/...` für Dokumentationsänderungen

## 2. Änderungen speichern (Committen)
Mache regelmäßig Commits mit aussagekräftigen Nachrichten.
```bash
git add .
git commit -m "Beschreibe kurz, was du gemacht hast"
```
*Tipp: Nutze klare, kurze Sätze (z.B. "Daten-Preprocessing für Task 1 hinzugefügt").*

## 3. Code auf GitHub hochladen (Pushen)
Wenn du fertig bist oder deinen Fortschritt sichern willst:
```bash
git push -u origin dein-branch-name
```

## 4. Pull Requests (PR) & Code Review
Wenn dein Code fertig für den `main`-Branch ist:
1. Gehe auf GitHub.
2. Erstelle einen **Pull Request (PR)** von deinem Branch in den `main`-Branch.
3. Bitte mindestens eine der anderen zwei Personen, den Code kurz anzuschauen (Code Review).
4. Wenn alles gut ist, klickt ihr auf **Merge PR**.

## 5. Konflikte vermeiden
- **Aufgaben aufteilen:** Sprecht euch ab! Einer macht z.B. Task 1, der andere Task 2. Wenn ihr in denselben Dateien arbeitet, kommt es zu "Merge-Konflikten".
- **Häufig updaten:** Wenn jemand anderes etwas in `main` gemerged hat, hole dir die Änderungen in deinen Branch:
  ```bash
  git checkout dein-branch-name
  git merge main
  # oder: git pull origin main
  ```

## Zusammenfassung des täglichen Workflows:
1. `git checkout main` & `git pull origin main` (Aktuellsten Stand holen)
2. `git checkout -b feature/mein-feature` (Neuen Branch erstellen)
3. *... Code schreiben ...*
4. `git add .` & `git commit -m "Mein Fortschritt"` (Änderungen speichern)
5. `git push -u origin feature/mein-feature` (Hochladen)
6. Pull Request auf GitHub erstellen & mergen lassen.
