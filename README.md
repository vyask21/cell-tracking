# Cell tracking during development

Tracking cells through 3D+time light-sheet microscopy of developing zebrafish, with
sparse ground truth: detect cell centres in every frame, link them across time, and
recover divisions.

- Competition: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
- Metric: adjusted edge Jaccard plus 0.1 times division Jaccard. Full definition in
  [`NOTES.md`](NOTES.md).
- Code competition. The scored test set is hidden and swapped in at rerun time, so
  inference has to run over roughly 85 GB inside a 12 h notebook with no internet
  access.
- Deadline: 2026-09-29
- Final placement: 98th of 4,017, silver medal.

## Result

The pipeline detects cell centres in 3D microscopy volumes, links them from frame to
frame, and decides where a cell has divided into two. It started from the strongest
public notebook in the competition and added two pieces of its own. The first is a
small network that nudges each detected centre by up to two microns, averaged with the
one that notebook shipped. The second replaces the hand-written rule that decided cell
divisions with a gradient-boosted model trained on division candidates from 100
training videos, which on its own lifted the private score from 0.921 to 0.929. The
submission that counted scored 0.928 privately, 98th of 4,017 teams, a silver medal.

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
