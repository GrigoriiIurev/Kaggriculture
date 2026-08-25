# Stage 3: League Market Training

Stage 3 keeps the submitted 1601-rated agent as an immutable incumbent. A small
policy may only control sales of `MILK`, `WOOL`, `STRAWBERRY`, and `MELON`; it
cannot change routes, workers, crops, animals, hires, land purchases, or the
incumbent's non-premium market orders.

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
opponent-farm changes over 1 and 24 turns. PPO acts only when a sale can actually
be changed. For each premium product it can keep the incumbent command or sell
25%, 50%, 75%, or 100%. It cannot suppress an incumbent sale or hold inventory
indefinitely. Only sales above the base price receive a small quality bonus;
cheap sales are not punished. Every non-fallback choice becomes a full
liquidation when the shed is nearly full or during the final three days.

The deterministic PPO policy starts as an exact copy of the incumbent decision:
all actor output weights are zero and every product selects `KEEP_INCUMBENT`.
Exploration happens only during stochastic training. Evaluation happens after
each short 10,000-decision round, and a catastrophic result restores the last
genuinely safe PPO checkpoint.

Every evaluation uses the same seeds and both seats for the unchanged incumbent
and the candidate. A candidate is promoted only if it has no runtime errors,
improves the overall score/margin ordering, and does not materially regress on
the Stage 2 veto opponents. The report also records every selected action and
the fraction that actually changed a market command, so a no-op policy can no
longer look like successful learning. Otherwise `submission.tar.gz` contains an
exact zero-residual wrapper around the incumbent, so failed training cannot
replace a known-working strategy.

## Colab

Open `kaggriculture_stage3_league_training_colab.ipynb`. Google Drive is mounted
in the first cell, and the long-running cell streams every output line both to
the notebook and to:

`MyDrive/Kaggriculture/results/stage3_league_market_v4/pipeline.log`

The final archive and reports are written to the same directory. Kaggle upload
is disabled by default; set `SUBMIT_TO_KAGGLE = True` only after reviewing the
promotion result.
