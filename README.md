# Final Project- COMP 550- NLP
###Team Mebers: 
####Ryan Languay, 
####Sogol Masoumzadeh, 
####Miguel l. Salinas. 

## Development
|Dependency | Version|
|------------|-------|
|Python | > 3.6|

Installing project requirements:

```bash
# Using a virtual environment
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

pip install -e .
```

Running modules:

```bash
#For running the corpus_preprocessor: Craeting the datasets 
using the raw data.

pyhton -m pre_processor
```


```bash
#For running the main engine: Running the experiments using 
the appropriate flags.

pyhton -m engine_model



