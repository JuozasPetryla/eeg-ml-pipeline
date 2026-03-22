# Psich.ai

## Setup

Go to readme of -> https://github.com/JuozasPetryla/eeg-infra

Additionally set a `venv` locally:

```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Usage

You can run scripts either normally on your local machine:

`python -m src.ml.statistics --local`

Or you can run scripts against local docker container:

`docker exec -it eeg-ml-pipeline python -m src.ml.statistics --job_id {int_job_id}`
