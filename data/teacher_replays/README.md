# Teacher replays

Download every available public replay from all active submissions of the current
top N players:

```bash
python3 download_top_replays.py --top-players 10
```

Preview the amount without downloading:

```bash
python3 download_top_replays.py --top-players 10 --list-only
```

To limit disk usage, keep only the newest 20 replays per player:

```bash
python3 download_top_replays.py --top-players 10 --max-replays-per-player 20
```

Use only the highest-scoring active submission of each player:

```bash
python3 download_top_replays.py --top-players 10 --best-submission-only
```

You can also place manually downloaded public Kaggriculture replay JSON files here.
Then run:

```bash
python3 build_teacher_dataset.py
```

These files are processed separately from the team's own Kaggle replays.
