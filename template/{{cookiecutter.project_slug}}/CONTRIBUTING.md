## Contributing to {{ cookiecutter.project_name }}

### Step 1. Fork this repository to your GitHub

### Step 2. Clone the repository from your GitHub

```sh
git clone https://github.com/[YOUR GITHUB ACCOUNT]/{{ cookiecutter.project_slug }}.git
```

### Step 3. Add upstream remote

```sh
git remote add upstream "https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}"
```

You can pull the latest code in main branch through `git pull upstream main` afterward.

### Step 4. Check out a branch for your new feature

```sh
git checkout -b [YOUR FEATURE]
```

### Step 5. Install prerequisites

This project uses conda + Poetry for environment management.

```sh
# Create the conda env if you haven't already
conda create -n {{ cookiecutter.project_slug }} python={{ cookiecutter.python_version_min }} -y
conda activate {{ cookiecutter.project_slug }}
POETRY_VIRTUALENVS_CREATE=false poetry install --with dev
pre-commit install
# Update hook versions to latest (do this periodically)
pre-commit autoupdate
```

> **Note:** The template ships with pinned pre-commit hook versions for reproducibility.
> Run `pre-commit autoupdate` after initial setup to get the latest versions,
> then commit the updated `.pre-commit-config.yaml`.

### Step 6. Work on your new feature

Commit messages **must** follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format. The `commitizen` pre-commit hook validates every commit message automatically.

```sh
git commit -m "feat(scope): description of change"
```

Common commit types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `style`, `ci`.

Alternatively, use `cz commit` for an interactive prompt:

```sh
cz commit
```

### Step 7. Run quality gates

Run the full pipeline (lint -> typecheck -> tests):

```sh
nox
```

Individual sessions:

```sh
nox -s lint       # Ruff linting + format check
nox -s typecheck  # mypy --strict on src, tests, noxfile.py
nox -s tests      # pytest full suite
```

### Step 8. Generate documentation

```sh
nox -s docs
```

### Step 9. Create a Pull Request and celebrate
