# deadlock-win-prediction

Vorhersage des Gewinnerteams in Deadlock mittels eines neuronalen Netzes. 
Das Modell analysiert einen Snapshot des aktuellen Spielzustands und sagt dessen Ausgang voraus.

## Usage of the project

### Setup of the project

1. Clone the project via ssh or https:
```
git clone https://github.com/NovaCaller/deadlock-win-prediction.git
git clone git@github.com:NovaCaller/deadlock-win-prediction.git
```
2. If not done yet, setup a _virtual environment_ (in the local project folder of the cloned repository):

```bash
python3 -m venv venv
```

3. Activate the virtual environment:

```bash
source venv/bin/activate
```

4. Install the dependencies

```bash
pip install -r requirements.txt
```

### Data Acquisition

Execute `pull_database_dump.py` to download the match data.

The script downloads relevant parquet-files from the `match_metadata`-folder of the official database-dump
(https://files.deadlock-api.com/buckets/db-snapshot/public/).The downloaded files contain match-data for the  
patch-duration from 25.10.2025 – 21.11.2025.

The progress of the download will be displayed as a progress bar in the terminal.

### Datenaufbereitung und Feature Engineering

Für die Datenaufbereitung führe `prepare_data.py` aus.

### Explorative Analyse

Diese Funktionen sind nicht notwendig für das Modell, doch helfen dabei einen Überblick über das Modell zu gewinnen.

Relevante Plots / Daten dazu können mit Ausführen von  `explorative_analysis.py` (dort entsprechend Funktionen ein/auskommentieren wenn notwendig)
oder mit Ausführen der Jupyter Notebooks (zu finden unter dem Ordner `Notebooks`) erhalten werden.

### Modell Training

Um schließlich das Modell zu trainieren, muss die Datei `train_model.py` ausgeführt werden. Idealerweise sollte das Ausführen
der python-Datei im Terminal geschehen (mit `python train_model.py`) da in IDEs der zu anzeigende Trainingsladebalken inkorrekt formattiert werden kann.

Es wird über das Training zudem über die Trainingsepochen hinweg ausgegeben, inwiefern der Trainings/Validierungsloss und die Trainings/Validierungsaccuracy ausfällt.

### Modell-Prediction

Für die Modell Prediction ist `predict.py` auszuführen. Aktuell muss dort in der Datei ein Gamestate manuell (fest im Code) eingegeben werden. Für eine spätere Version
ist aber geplant, hierfür eine UI einzurichten, mit der die Eingabe leichter erfolgen kann.