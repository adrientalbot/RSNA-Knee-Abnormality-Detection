# RSNA Knee Abnormality Detection

Notebook-based experiments for the RSNA knee abnormality detection challenge.
The local environment is intentionally scoped to `notebooks/playground.ipynb`; the
baseline notebook's heavier dependencies are not installed.

## Local setup

[UV](https://docs.astral.sh/uv/) manages Python, the virtual environment, and
the dependency lockfile.

```bash
make setup
make notebook
```

`make setup` creates `.venv` and installs the dependencies from `uv.lock`.
To use the environment from another Jupyter installation, run `make kernel`
once and select **Python (RSNA Knee)**.

Useful commands:

```bash
make help      # list commands
make sync      # reproduce the locked environment
make labels    # generate labels for train.csv with JEV
make clean     # remove generated caches
```

## Generate JEV labels

The labeler starts with `report_labels_v2.csv`, preserves existing human labels,
and asks JEV only about unresolved `UNK` targets. JEV returns explicit
`YES`/`NO`/`UNK` choices. Final targets are old-style 0–1 soft scores: existing
v2 scores are preserved, while JEV scores are the probability-weighted average
of the `NO=0.08`, `UNK=0.28`, and `YES=0.82` anchors. Every successful response
is checkpointed, including the model confidence and all three choice
probabilities. Set the API key in your environment and run:

```bash
make labels
```

For a small trial before processing the full dataset:

```bash
make labels LABEL_ARGS="--limit 10"
```

Progress is saved to `artifacts/jev_choice_checkpoint.jsonl`, so rerunning the command
continues where it stopped. The completed dataset is written to
`artifacts/train_jev_choice_labeled.csv`. The `artifacts/` directory, `train.csv`, and
`.env` files are ignored by Git.

To audit JEV against already-resolved human and v2 labels without overwriting
them, add `--audit-all` through `LABEL_ARGS`.

The notebooks were developed in Kaggle and may import `kaggle_secrets`, which
is provided by the Kaggle runtime rather than this local environment. Never
commit API keys or local `.env` files.
