[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg?style=flat-square)](https://conventionalcommits.org)
[![Github Actions](https://github.com/dhilgart/NCAA_eval/actions/workflows/python-check.yaml/badge.svg)](https://github.com/dhilgart/NCAA_eval/actions/workflows/python-check.yaml)


# ncaa_eval

Evaluator for self-evaluation of NCAA March Mania Kaggle Competition models

## Getting Started

### Prerequisites
* [Python 3.12+](https://www.python.org/downloads/)
* [Poetry](https://python-poetry.org/docs/#installation) (dependency management)
* **Kaggle API credentials** (required for data sync — see below)

### Kaggle API Authentication

The `sync.py` data pipeline downloads NCAA data from the [Kaggle March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2025) competition. This requires a Kaggle account and API credentials.

**Setup:**

1. Create a free [Kaggle account](https://www.kaggle.com/account/login) if you don't have one.
2. Accept the competition rules at the [March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025) competition page.
3. Generate an API token: go to **Account → Settings → API → Create New Token**. This downloads `kaggle.json`.
4. Place it at `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows), then restrict permissions:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
   Alternatively, set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

For full details see the [Kaggle API documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md).

### Installation

```bash
poetry install
```

## Usage

### Sync Data

Fetch NCAA data from Kaggle and ESPN into a local Parquet store:

```bash
# Fetch all sources (Kaggle historical data + ESPN current season)
python sync.py --source all --dest data/

# Fetch Kaggle only (historical seasons)
python sync.py --source kaggle --dest data/

# Fetch ESPN only (requires Kaggle sync to have run first)
python sync.py --source espn --dest data/

# Bypass cache and re-download everything
python sync.py --source all --dest data/ --force-refresh
```

Subsequent runs skip already-cached Parquet files automatically.

## Contributing
See [Contributing](contributing.md)

## Authors
Dan Hilgart <dhilgart@gmail.com>

Created from [Lee-W/cookiecutter-python-template](https://github.com/Lee-W/cookiecutter-python-template/tree/1.11.0) version 1.11.0
