# Changelog

## 0.10.0 (2026-03-13)

### Feat

- **tutorial**: Ensemble Tutorial Notebook — Story 10.4 (#81)
- **dashboard**: ensemble models in dashboard leaderboard, deep dive, and bracket visualizer (Story 10.3) (#80)
- **ensemble**: Ensemble Inference Interface — predict_proba, predict_bracket, EnsembleProvider (Story 10.2) (#79)
- **model**: StackedEnsemble class and OOF training pipeline (Story 10.1) (#78)
- **ingest**: add Pandera schema validation to KaggleConnector (Story 9.14) (#76)
- **dashboard**: implement st.progress for Monte Carlo simulation (Story 9.12) (#74)
- **evaluation**: add custom metric plugin registry (Story 9.10) (#71)
- **cli**: add predict command for win-probability CSV generation (Story 9.9) (#70)
- **dashboard**: Story 9.8 — User-Editable Bracket with Override Cascade (#69)
- **evaluation**: add game-theory slider perturbation (Story 9.7) (#68)
- **ingest**: add post-sync data validation (Story 9.5) (#66)
- **model**: add feature importance for Elo and LogReg models (Story 9.3) (#64)
- **model**: embed FeatureConfig as model-level concern (Story 9.2) (#63)
- **evaluation**: add Kaggle submission export — CLI + dashboard (Story 9.1) (#62)

### Fix

- **dashboard**: Story 8.8 — Dashboard UX Quick Fixes (#56)
- **type-safety**: Story 8.6 — Type Safety & Configuration Improvements (#54)
- **ingest**: add ESPN retry logic, decouple SyncEngine from Typer, generalize dedup (Story 8.3) (#50)

### Refactor

- **tests**: consolidate duplicated test helpers into shared conftest (Story 9.13) (#75)
- **dashboard**: replace undocumented Streamlit selection API (Story 9.11) (#73)
- **api**: expose public APIs, eliminate private attribute access (Story 8.2) (#49)

## 0.9.0 (2026-02-25)

### Feat

- **dashboard**: Pool Scorer page with MC outcome analysis and CSV export (#42)

## 0.8.0 (2026-02-25)

### Feat

- **dashboard**: Build Presentation Page — Bracket Visualizer (Story 7.5) (#41)

## 0.7.0 (2026-02-24)

### Feat

- **dashboard**: Model Deep Dive page with reliability diagrams and feature importance (#40)

## 0.6.0 (2026-02-24)

### Feat

- **epic-7**: Build Lab Page — Backtest Leaderboard (Story 7.3) (#39)

## 0.5.0 (2026-02-23)

### Feat

- **model**: define Model ABC and plugin registry — story 5.2 (#27)

## 0.4.0 (2026-02-22)

### Feat

- **transform**: Elo feature building block with walk-forward temporal safety (#24)

## 0.3.0 (2026-02-22)

### Feat

- **transform**: implement stateful feature serving layer (Story 4.7) (#23)

## 0.2.0 (2026-02-19)

### Feat

- **ingest**: implement Kaggle and ESPN data source connectors (Story 2.3) (#12)
- **toolchain**: configure versioning, packaging & documentation (#7)
- **testing**: configure Hypothesis, Mutmut, and test framework (#5)
- **toolchain**: configure Ruff/Mypy/Pytest pre-commit hooks (#4)
- **workflow**: update code-review to generate PRs using template format
- **testing**: add fuzz-based testing approach to testing strategy
- **quality**: add pure functions vs side effects design guidance
- **quality**: add SOLID principles for testability
- **quality**: add PEP 20 compliance checks and complexity gates
- **workflow**: automate branch creation, atomic commits, and PR generation
- add template learning capture system and Epic 8
- initialize Poetry project with src layout and strict type checking

### Fix

- **testing**: resolve 6 critical and medium issues from code review
- **testing**: align markers across docs and add missing pyproject.toml config
