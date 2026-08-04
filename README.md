# biohub-cell-tracking-during-development



- Competition: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
- Metric:
- Deadline: 2026-09-29
- Final placement:

## Result

One paragraph, written at the end: what was built, what it scored, where it placed.

## Approach

What the validation scheme was and why, the features that mattered, the model, and
the one or two decisions that made the difference.

## Reproducing

```bash
pip install -r requirements.txt
python -m kaggle competitions download -c biohub-cell-tracking-during-development -p data/raw --unzip
python -m src.train --config conf/baseline.yaml
```

## Experiment log

Every run that was made, including the ones that failed, is in
[`experiments.csv`](experiments.csv). The reasoning is in [`NOTES.md`](NOTES.md).

```bash
python -m src.ledger      # prints the ledger, best CV first
```
