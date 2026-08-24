# Opponent League (Stage 2)

Stage 2 turns the selected public notebooks into a reproducible local test
league. It does not train or submit an agent.

## What it does

1. Downloads the six public notebook sources.
2. Recovers the final packaged `main.py` without executing notebook cells.
3. Verifies that every file compiles, exposes `agent`, and matches its pinned SHA-256.
4. Plays every requested match in both seats on the same seeds.
5. Saves each game immediately, so a Colab restart does not lose completed work.
6. Produces Bradley-Terry ratings, pair records, a veto set, and a promotion gate.

The notebook sources play different roles:

- `deniz_v111`: 8C/4S economic core;
- `ray_c95`: latest C95 artifact in the public findings notebook;
- `boatlee_r5a`: bounded recovery variant;
- `kaito_v43`: sparse-shop daily market hybrid;
- `bruce_route1`: an alternative high-score route;
- `ray_k320`: the packaged K320 agent from the ranking notebook.

## Colab

Open `kaggriculture_opponent_league_colab.ipynb`. The first code cell mounts
Google Drive before any long job starts.

Default mode evaluates only the current challenger against all six opponents:

```python
FULL_ROUND_ROBIN = False
SEED_COUNT = 2
```

That is 24 games: 6 opponents x 2 seeds x 2 seats. Set
`FULL_ROUND_ROBIN = True` to rank every pair. With one challenger, six public
agents, and two seeds, that is 84 games.

All durable files are written to:

```text
MyDrive/Kaggriculture/league/
```

The important outputs are:

- `opponent_pool.json`: paths, hashes, ranking, sampling weights, and veto opponents;
- `results/games.jsonl`: one checkpointed record per game;
- `results/pair_results.csv`: pairwise records;
- `results/rankings.csv`: Bradley-Terry table;
- `results/report.md`: readable summary.
- `league_pipeline.log`: complete live log, including a traceback if a run fails.

Rerunning the same configuration reuses completed games. A changed agent file
has a changed SHA-256 key and is evaluated again automatically.

## Local command

```bash
python3 -u run_league_pipeline.py \
  --drive-root ./artifacts/league_run \
  --challenger submission.tar.gz \
  --challenger-name our_agent \
  --seed-count 2
```

Use `--max-games 1` for a quick pipeline check. Use `--full-round-robin` for
the complete league.

## How to read the gate

The initial gate requires:

- every requested game is complete;
- zero challenger runtime errors;
- challenger score rate is at least 50% across this pool.

This gate is intentionally simple. Stage 3 will compare a trained candidate
against both the current incumbent and the three strongest veto opponents, so
an improvement cannot be promoted by exploiting one weak bot.
