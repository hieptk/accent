# ACCENT: Counterfactual Explanations for Neural Recommenders
## Environment
Python 3.7
Tensorflow 2.2.0

## Running Full Experiment
```bash
python experiment.py --algo ALGO
```
where *ALGO* indicates the explanation algorithm: "attention", "pure_att", "fia", "pure_fia", "accent".
Results will be stored in CSV files: *{ALGO}_{k}.csv* with k = 5, 10, 20.
