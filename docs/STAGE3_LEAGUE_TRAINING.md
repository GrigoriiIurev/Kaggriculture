# Stage 3: League Market Training

Stage 3 keeps the submitted 1601-rated agent as an immutable incumbent. A small
policy may only append sales of `MILK`, `WOOL`, `STRAWBERRY`, and `MELON`; it
cannot change routes, workers, crops, animals, hires, land purchases, or the
incumbent's existing market orders.

## Inputs from earlier stages

- `MyDrive/Kaggriculture/league/opponent_pool.json` from Stage 2 supplies the six
  opponents, sampling weights, and veto set.
- `MyDrive/Kaggriculture/results/rl_boatlee_market_v2/submission.tar.gz` is the
  unchanged incumbent.
- Stage 1 replay analysis is copied into `replay_context.json` when present. It is
  diagnostic context, not a source of hidden game state.

## Training and gate

The observation includes current public state, the private inventory available
to the agent, market price/inventory changes over 1, 4, and 24 turns, and public
opponent-farm changes over 1 and 24 turns. PPO chooses one of four sale fractions
for each premium product: 0%, 25%, 50%, or 100%.

Every evaluation uses the same seeds and both seats for the unchanged incumbent
and the candidate. A candidate is promoted only if it has no runtime errors,
improves the overall score/margin ordering, and does not materially regress on
the Stage 2 veto opponents. Otherwise `submission.tar.gz` contains an exact
zero-residual wrapper around the incumbent, so failed training cannot replace a
known-working strategy.

## Colab

Open `kaggriculture_stage3_league_training_colab.ipynb`. Google Drive is mounted
in the first cell, and the long-running cell streams every output line both to
the notebook and to:

`MyDrive/Kaggriculture/results/stage3_league_market/pipeline.log`

The final archive and reports are written to the same directory. Kaggle upload
is disabled by default; set `SUBMIT_TO_KAGGLE = True` only after reviewing the
promotion result.
