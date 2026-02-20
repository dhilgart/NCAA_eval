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
2. **Verify your phone number** — Kaggle requires phone verification before the API can access any competition data. Go to **Account → Settings → Phone Verification** and complete verification.
3. **Accept the competition rules** — required for each competition year you want data from:
   - [March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/rules) (historical data through the 2025 tournament)
   - Accept the current year's competition rules as well if it is listed on [Kaggle](https://www.kaggle.com/competitions?search=march+machine+learning+mania)
4. Get your API token: go to **Account → Settings → API → Create New Token** and copy the token value (starts with `KGAT_`).
5. Save the token to `~/.kaggle/access_token`:
   ```bash
   mkdir -p ~/.kaggle
   echo "KGAT_your_token_here" > ~/.kaggle/access_token
   chmod 600 ~/.kaggle/access_token
   ```

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
