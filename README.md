# deadlock-win-prediction

Prediction of the winner team in Deadlock with the use of a neuronal network . 
The model then analyses a snapshot of the current game state und predicts its outcome.

## Usage of the project

### Setup of the project

1. Clone the project via ssh or https:
```
git clone https://github.com/NovaCaller/deadlock-win-prediction.git
git clone git@github.com:NovaCaller/deadlock-win-prediction.git
```
2. If not done yet, setup a _virtual environment_ (in the local project folder of the cloned repository):

```bash
cd deadlock-win-prediction
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

note that `torch` is used as a dependency, but it is deliberately not included in `requirements.txt`, since the correct torch package for your system will be installed during runtime.

### Data Acquisition

Execute `pull_database_dump.py` to download the match data.

The script downloads relevant parquet-files from the `match_metadata`-folder of the official database-dump
(https://files.deadlock-api.com/buckets/db-snapshot/public/).The downloaded files contain match-data for the  
patch-duration from 2025-10-25 to 2025-11-21.

The progress of the download will be displayed as a progress bar in the terminal.

### Data Preparation and Feature Engineering

For the data preparation execute `prepare_data.py`.

### Explorative Analysis

The functions are not required for the model in itself, but they help to get an overview over the data in general.

Relevant plots and data can be acquired with the execution of `explorative_analysis.py` (there, you can comment out/in functions if you do(or not) need them)
or you can execute the Jupyter Notebooks (to be found under the folder `Notebooks`) to also get them. 

### Model Training

To ultimately train the model, execute the python file `train_model.py`. Ideally, the execution should be done via command prompt, in the terminal with `python  train_model.py` since
the training-loading bar may be incorrectly formatted if executed within the IDE.

Over the training, there will be output throughout the training epochs. For example, that information includes the training and validation loss, as well as the respective accuracy (within that epoch).

### Model-Prediction

For the model prediction execute `predict.py`. For now, in this file, the to-be-predicted game state has to be manually typed in (in the code, that is.). 

For a later version of this project, it is planned to instead make us of an UI to make this task a bit easier and less arduous.