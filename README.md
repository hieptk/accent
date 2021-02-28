# ACCENT: Counterfactual Explanations for Neural Recommenders
## Environment
- Python 3.7
- Tensorflow 2.2.0

## Training Models
### RCF
```bash
cd RCF
python train.py --pretrain -1
```

### NCF
```bash
cd NCF/src/scripts
python train.py
```

## Running Experiment
### RCF
```bash
cd RCF
python experiment.py --algo ALGO
```

### NCF
```bash
cd NCF/src/scripts
python experiment.py --algo ALGO
```
where *ALGO* indicates the explanation algorithm: "attention", "pure_att", "fia", "pure_fia", "accent".
Results will be stored in CSV files: *{ALGO}_{k}.csv* with k = 5, 10, 20.
