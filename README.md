# Cell tracking during development

My entry to the Biohub cell tracking competition on Kaggle. The task is to follow
every cell in 3D light-sheet movies of a developing zebrafish embryo: find the
centre of each cell in each frame, link each cell to itself in the next frame, and
mark the moments when a cell divides into two.

- Competition: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
- Final placement: 98th of 4,017 teams, silver medal.

## Result

The pipeline is built on public Kaggle notebooks, the best of which scored 0.953 on
the public leaderboard and 0.917 on the private one. Two additions of my own took
it to 0.929 private. The first is a small network that moves each
detected cell centre by up to two microns, averaged with the one that notebook
used. The second replaces the hand-written rule that decided where cells divide
with a gradient-boosted model trained on division candidates from 100 training
videos. The submission that counted scored 0.928 privately and placed 98th of
4,017 teams, a silver medal.

## How the score works

The score adds two parts. The first is an edge score: the share of true
frame-to-frame links the prediction gets right, scaled down if the prediction has
more cells than the organisers' estimate. The second is 0.1 times a division
score, the same kind of overlap measured on cell divisions only. The ground truth
is sparse: only some cells in each movie are annotated, and predictions elsewhere
are neither rewarded nor penalised except through the cell count.

The strongest public notebooks scored about 0.93 on edges and only 0.12 to 0.23 on
divisions. Divisions were the weakest part of every public pipeline, and most of
my final gain came from them.

## Approach

### Validation

The public model weights were trained on 180 of the 199 training movies. Any score
measured on those 180 is inflated, so the 19 movies left out of that training are
the only local test that behaves like the hidden test set. The local numbers I
relied on most were measured on those 19.

For much of the competition the local test and the public leaderboard disagreed
about changes to decision thresholds, so threshold changes were settled by
submitting them. The division classifier was the exception: its estimate on the
19 held-out movies, about +0.01, came close to the private result of +0.008.

### My own pipeline first

The first month went into a pipeline written from scratch in `src/`: a detector,
linking by assignment and later by integer linear programming, graph clean-up
rules, and a rule for adding divisions. It reached 0.892 public and 0.879 private.
Its code, tests and every configuration are in this repo.

### The public stack, and the coordinate head

By mid-September a public notebook scored 0.947 and hundreds of teams had copied
it. I used it as the base from then on and kept my changes one at a time against
it. A later public notebook added a small network that shifts each detected
centre, and its author kept the trained weights private at first. I captured detections and their matched ground truth on training movies
and fitted my own version. Mine scored lower on its own, but averaging my head's
shift with the published one scored 0.956 public, above either alone.

### A division classifier

The public stack decided divisions with fixed distance gates, a nearest-neighbour
check and a veto from a second network. I ran the pipeline over 119 training
movies, logged every division it considered under looser gates with 19
measurements each, and labelled them against the ground truth. A LightGBM model
trained on those candidates replaced the fixed rule. It ranked held-out candidates
at an AUC of 0.99. On the private leaderboard it moved the score from 0.921 to
0.929, the largest single gain of the competition.

## What did not work

- Tuning single settings on the public notebook. Its author had tuned it against
  the leaderboard, and almost every change I made to one setting lost.
- Choosing coordinate heads by how close they moved centres to the ground truth.
  That measure ranked three heads in the opposite order to the leaderboard.
- A retrained division model at a stricter threshold. It had the best public score
  of the final week, 0.958, and one of the worst private scores, 0.923.

## What is in this repo

- `src/`: the pipeline, importable from a Kaggle notebook, with tests in `tests/`.
- `conf/`: one configuration per experiment.
- `scripts/`: the screens, the capture and labelling kernels, the fitting code for
  the coordinate heads and the division model, and the builders that produced
  each late submission.
- `experiments.csv`: every experiment, with its reasoning, its local numbers and
  both leaderboard scores.

The notebooks derived from other people's public Kaggle notebooks are not
included, because their licences require attribution that this repo does not
carry. The builders in `scripts/` show each change made to them.

## Running it

The competition data is about 100 GB of OME-Zarr and is not redistributed here.
Training and inference ran in Kaggle notebooks, since this work was done on a
machine without a GPU.

```bash
pip install -r requirements.txt
python -m pytest -q
python -m src.ledger
```

## Licence

MIT, see [LICENSE](LICENSE). Covers the code in this repo only.
