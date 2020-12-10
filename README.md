# Final Project- COMP 550- NLP
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
#For running the main engine without hand-crafted features:

pyhton -m engine_model
```
```bash
#For running the main engine including hand-crafted features:

python -m spacy download en_core_web_sm
```


