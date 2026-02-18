## Contributing to NCAA Eval

### Step 1. Fork this repository to your GitHub

### Step 2. Clone the repository from your GitHub

```sh
git clone https://github.com/[YOUR GITHUB ACCOUNT]/NCAA_eval.git
```

### Step 3. Add upstream remote

```sh
git remote add upstream "https://github.com/dhilgart/NCAA_eval"
```

You can pull the latest code in main branch through `git pull upstream main` afterward.

### Step 4. Check out a branch for your new feature

```sh
git checkout -b [YOUR FEATURE]
```

### Step 5. Install prerequisites

This project uses conda + Poetry for environment management.

```sh
conda activate ncaa_eval
POETRY_VIRTUALENVS_CREATE=false poetry install --with dev
pre-commit install
```

### Step 6. Work on your new feature

Commit messages **must** follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format. The `commitizen` pre-commit hook validates every commit message automatically and will reject non-conforming messages.

```sh
git commit -m "feat(scope): description of change"
```

Common commit types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `style`, `ci`.

**GitHub PRs note:** When merging via GitHub squash-merge, the PR title is used as the final commit message. PR titles must also follow conventional commits format (e.g., `feat(scope): description`) for `cz check` to pass on the main branch history.

Alternatively, use `cz commit` for an interactive prompt that guides you through the format:

```sh
cz commit
```

### Step 7. Run quality gates

Run the full pipeline (lint → typecheck → tests):

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

This runs `sphinx-apidoc` to regenerate API stubs, then `sphinx-build` to produce HTML in `docs/_build/html/`.

### Step 9. Preview version bump (optional)

To see what version bump will be applied based on your commits:

```sh
cz bump --dry-run --yes
```

Expected output: `bump: version X.Y.Z → A.B.C` with the detected increment type.

### Step 10. Create a Pull Request and celebrate 🎉
