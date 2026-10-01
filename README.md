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
make clean     # remove generated caches
```

The notebooks were developed in Kaggle and may import `kaggle_secrets`, which
is provided by the Kaggle runtime rather than this local environment. Never
commit API keys or local `.env` files.
