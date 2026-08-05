# Cell tracking during development

Tracking cells through 3D+time light-sheet microscopy of developing zebrafish, with
sparse ground truth: detect cell centres in every frame, link them across time, and
recover divisions.

- Competition: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
- Metric: adjusted edge Jaccard plus 0.1 times division Jaccard. Full definition in
  [`NOTES.md`](NOTES.md).
- Code competition. The scored test set is hidden, embryo-disjoint from train, and
  swapped in at rerun time, so inference has to run over roughly 85 GB inside a 12 h
  notebook with no internet access.
- Deadline: 2026-09-29
- Final placement:

## Result

One paragraph, written at the end: what was built, what it scored, where it placed.

## Approach

What the validation scheme was and why, the features that mattered, the model, and
the one or two decisions that made the difference.

## Reproducing

The competition data is roughly 100 GB of OME-Zarr, so there is no single download
line that is honest here. See [`NOTES.md`](NOTES.md) for what is held locally and
what is read straight from the Kaggle mount.

```bash
pip install -r requirements.txt
```

## Experiment log

Every run that was made, including the ones that failed, is in
[`experiments.csv`](experiments.csv). The reasoning is in [`NOTES.md`](NOTES.md).

```bash
python -m src.ledger      # prints the ledger, best CV first
```
