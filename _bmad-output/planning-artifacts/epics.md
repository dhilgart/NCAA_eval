---
stepsCompleted:
  - step-01-validate-prerequisites
  - step-02-design-epics
  - step-03-create-stories
inputDocuments:
  - specs/03-prd.md
  - specs/05-architecture-fullstack.md
  - specs/04-front-end-spec.md
---

# NCAA_eval - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for NCAA_eval, decomposing the requirements from the PRD, UX Design if it exists, and Architecture requirements into implementable stories.

## Requirements Inventory

### Functional Requirements

FR1 (Unified Data Ingestion): The system must ingest, clean, and standardize raw NCAA data from multiple external sources into a unified internal schema.
FR2 (Persistent Local Store): The system acts as a Single-User Data Warehouse. It must support a "One-Time Sync" command that fetches historical data and persists it locally (e.g., Parquet/SQLite). This local store acts as the authoritative Source of Truth for all downstream training and evaluation.
FR3 (Smart Caching): The ingestion engine must implement a caching layer that strictly prefers valid local data over remote API calls to minimize latency and rate-limiting.
FR4 (Chronological Serving): The Data API must support strict chronological streaming `get_chronological_season(year)` to support "walk-forward" training and prevent data leakage.
FR5 (Advanced Transformations): The platform must provide a library of transformations for: Sequential Features (rolling averages, streaks, momentum), Opponent Adjustments (linear algebra solvers for efficiency stats), Graph Representations (NetworkX graph objects for centrality metrics), and Normalization (canonical mapping of diverse team names to single IDs).
FR6 (Flexible Model Contract): The system must provide an abstract base class (`Model`) that supports: Stateless Models (standard batch training, e.g., XGBoost) and Stateful Models (models maintaining internal state across a season, e.g., Elo ratings).
FR7 (Hybrid Evaluation Engine): The evaluation system must calculate: Probabilistic Metrics (Log Loss, Brier Score, ROC-AUC), Calibration Metrics (ECE and reliability diagrams), and Tournament Scoring (user-defined point schedules applied to simulated brackets).
FR8 (Validation Workflow): The system must support "Leave-One-Tournament-Out" backtesting with strict temporal boundaries. Must gracefully handle the 2020 "COVID Year" by allowing models to update state (training) without attempting to evaluate predictions (testing).
FR9 (Monte Carlo Tournament Simulator): The system must implement a simulation engine capable of generating N (configurable, default 10,000) realizations of the tournament bracket based on a model's probability matrix to calculate "Expected Points" and "Bracket Distribution" metrics.

### NonFunctional Requirements

NFR1 (Performance - Vectorization): All core metric calculations must use vectorized operations (e.g., numpy) to minimize overhead during expensive cross-validation loops.
NFR2 (Performance - Parallelism): The system must support parallel execution of cross-validation folds and model evaluations to maximize throughput on multi-core systems.
NFR3 (Extensibility): The system must utilize a plugin-registry architecture to allow users to inject custom metrics, scoring functions, and feature generators without modifying core code.
NFR4 (Reliability - Leakage Prevention): APIs must be architected to strictly enforce temporal boundaries, making it impossible for a model to access future game data during training.
NFR5 (Reliability - Fail-Fast Debugging): The system must provide deep logging, error traces, and data assertions to facilitate debugging. Custom verbosity levels must be supported.

### Additional Requirements

**From Architecture:**
- Greenfield project initialized from scratch using standard Python package structure managed by Poetry (src layout)
- Repository Pattern: Abstracts data access (SQL/Parquet) behind a consistent API, decoupling business logic from storage mechanism
- Strategy Pattern: Used for Model ABC, allowing swapping between Stateful/Stateless models without changing evaluation pipeline
- Monolithic Package: All logic encapsulated in a single installable library (`ncaa_eval`)
- Local Storage: Either SQLite or Parquet (decision deferred); must serve as single authoritative data store
- Data structures between Logic and UI must use Pydantic models or TypedDicts (type sharing)
- Dashboard must never read files directly; must call `ncaa_eval` functions (No Direct IO in UI)
- `mypy --strict` compliance is mandatory
- Vectorization First: Reject PRs that use `for` loops over Pandas DataFrames for metric calculations
- Input Validation: Configuration files (JSON/YAML) validated via Pydantic
- Temporal boundary enforcement in API to prevent data leakage
- Development workflow: Poetry install -> Nox (Ruff + Mypy + Pytest) -> CLI training -> Streamlit dashboard
- Data entities defined: Team, Game, Season, ModelRun, Prediction, TournamentBracket

**From UX Spec:**
- Dark Mode enforced by default to reduce eye strain during long analysis sessions
- Desktop Only (Wide Mode) — mobile is not supported
- Interaction response must be under 500ms for diagnostic plots and bracket updates
- Heavy use of `@st.cache_data` for historical datasets and model artifacts
- Monospace fonts (IBM Plex Mono or system default) for all data tables and code snippets
- Functional color palette: Green (#28a745) for improvement, Red (#dc3545) for regression, Neutral (#6c757d) for structural
- Custom Streamlit Component (D3.js or Mermaid.js wrapper) for interactive bracket tree with clickable nodes
- ROI Simulations (10k+ iterations) should run asynchronously or provide progress bar (`st.progress`)
- Streamlit Multipage App with persistent sidebar navigation and `st.session_state` for filters
- Diagnostic Cards using `st.metric`, Heatmap DataFrames with Pandas conditional styling
- Simulation Sliders in sidebar for Game Theory inputs (Upset Aggression, Chalk Bias, Seed-Weight)

### FR Coverage Map

| Requirement | Epic | Description |
|:---|:---|:---|
| FR1 | Epic 2 | Unified data ingestion from multiple sources |
| FR2 | Epic 2 | Persistent local store as Source of Truth |
| FR3 | Epic 2 | Smart caching preferring local data |
| FR4 | Epic 4 | Chronological serving for walk-forward training |
| FR5 | Epic 4 | Advanced transformations (sequential, graph, opponent adj, normalization) |
| FR6 | Epic 5 | Flexible Model ABC (stateless + stateful) |
| FR7 | Epic 6 | Hybrid evaluation engine (probabilistic, calibration, tournament scoring) |
| FR8 | Epic 6 | Validation workflow with temporal boundaries + COVID handling |
| FR9 | Epic 6 | Monte Carlo tournament simulator |
| NFR1 | Epic 6 | Vectorized metric calculations |
| NFR2 | Epic 6 | Parallel cross-validation execution |
| NFR3 | Epic 5 | Plugin-registry extensibility |
| NFR4 | Epic 4 | Temporal boundary enforcement |
| NFR5 | Epic 1 | Fail-fast debugging toolchain + runtime logging & assertions |
| UI-3 | Epic 7 | Jupyter progress bars (Story 7.1) |
| UI-10 | Epic 7 | Comprehensive user guide (Story 7.8) |
| UI-11 | Epic 7 | Step-by-step tutorials (Story 7.9) |

## Epic List

### Epic 1: Project Foundation & Developer Toolchain
Developer can clone, install, lint, type-check, test, and commit against a fully configured Python project with enforced quality gates and runtime debugging infrastructure.
**FRs covered:** NFR5 (Fail-Fast Debugging via toolchain + runtime logging & assertions)

### Epic 2: Data Ingestion & Local Warehouse
User can fetch NCAA data from external sources, persist it locally, and access it with smart caching -- the "Source of Truth" is operational.
**FRs covered:** FR1, FR2, FR3

### Epic 3: Exploratory Data Analysis
User can explore ingested data to understand quality, structure, and relationships, producing documented findings that inform feature engineering.
**FRs covered:** Research enabler informing FR5

### Epic 4: Feature Engineering Suite
User can transform raw game data into ML-ready features including sequential stats, opponent adjustments, graph centrality, and canonical team IDs, with chronological data serving that enforces temporal boundaries.
**FRs covered:** FR4, FR5 | NFR4 (Leakage Prevention)

### Epic 5: Core Modeling Framework
User can train, predict, and persist models using a standardized contract that supports both stateful (Elo) and stateless (XGBoost) approaches.
**FRs covered:** FR6 | NFR3 (Extensibility)

### Epic 6: Evaluation & Validation Engine
User can evaluate models with probabilistic metrics, calibration analysis, walk-forward cross-validation, and Monte Carlo tournament simulation.
**FRs covered:** FR7, FR8, FR9 | NFR1 (Vectorization), NFR2 (Parallelism)

### Epic 7: Lab & Presentation Dashboard
User can visualize model performance via interactive Streamlit dashboards including leaderboards, reliability diagrams, bracket visualizer, point outcome analysis, and comprehensive documentation.
**FRs covered:** UI requirements from PRD Section 3 + UX Spec + UI-10 (User Guide) + UI-11 (Tutorials)

## Epic 1: Project Foundation & Developer Toolchain

Developer can clone, install, lint, type-check, test, and commit against a fully configured Python project with enforced quality gates.

### Story 1.1: Initialize Repository & Package Structure

As a developer,
I want a Poetry-managed Python project with src layout and core directory scaffolding,
So that I can `poetry install` into a working virtualenv with the correct package structure.

**Acceptance Criteria:**

**Given** a fresh clone of the repository
**When** the developer runs `poetry install`
**Then** a virtualenv is created with all core dependencies installed
**And** the `src/ncaa_eval/` package is importable (`import ncaa_eval` succeeds)
**And** the directory structure matches the Architecture spec: `src/ncaa_eval/{ingest,transform,model,evaluation,utils}/`, `dashboard/`, `tests/`, `data/`
**And** `pyproject.toml` specifies Python 3.12+ and declares all PRD-required dependencies (pandas, numpy, xgboost, scikit-learn, networkx, joblib, plotly, streamlit)
**And** a `.gitignore` excludes `data/`, virtualenvs, and common Python artifacts

### Story 1.2: Define Code Quality Standards & Style Guide

As a developer,
I want documented decisions on docstring convention, naming standards, import ordering, and PR checklist requirements,
So that all contributors follow consistent patterns and code reviews have clear criteria.

**Acceptance Criteria:**

**Given** the project needs a style guide before tooling is configured
**When** the developer reads the documented standards
**Then** the guide specifies the chosen docstring convention (numpy vs google style) with rationale
**And** naming conventions for modules, classes, functions, and variables are defined
**And** import ordering rules are specified (stdlib, third-party, local)
**And** a PR checklist is defined covering: type-check pass, lint pass, test pass, docstring coverage, and review criteria
**And** the "Vectorization First" rule is documented (no `for` loops over DataFrames for metric calculations)
**And** the guide is committed as a project document accessible to all developers

### Story 1.3: Define Testing Strategy

As a developer,
I want documented decisions on test types, coverage targets, and which checks run at pre-commit vs. PR-time,
So that I know what tests to write and when they'll be executed.

**Acceptance Criteria:**

**Given** the project needs a testing strategy before test tooling is configured
**When** the developer reads the documented strategy
**Then** it defines when to use each test type: unit, integration, property-based (Hypothesis), and mutation (Mutmut)
**And** it specifies coverage targets and whether coverage gates are enforced
**And** it defines the pre-commit check suite (fast checks: lint, type-check, fast unit tests)
**And** it defines the PR-time check suite (full checks: all tests, mutation testing, coverage report)
**And** it documents fixture conventions and test file organization (`tests/unit/`, `tests/integration/`, etc.)
**And** it provides guidance on when to use Hypothesis property-based tests vs. standard parametrized Pytest tests
**And** the strategy is committed as a project document accessible to all developers

### Story 1.4: Configure Code Quality Toolchain

As a developer,
I want Ruff, Mypy, and pre-commit hooks configured to enforce the agreed standards,
So that every commit is automatically checked for style, formatting, and type correctness.

**Acceptance Criteria:**

**Given** the code quality standards from Story 1.2 are documented
**When** the developer runs `pre-commit run --all-files`
**Then** Ruff checks and auto-fixes formatting and linting rules matching the agreed style guide
**And** Mypy runs in `--strict` mode and reports type errors
**And** pre-commit hooks are defined in `.pre-commit-config.yaml` and run automatically on `git commit`
**And** Ruff configuration in `pyproject.toml` enforces the chosen docstring convention and import ordering
**And** a developer introducing a type error or style violation is blocked from committing

### Story 1.5: Configure Testing Framework

As a developer,
I want Pytest, Hypothesis, and Mutmut configured with the agreed testing strategy,
So that I can run tests locally and CI enforces the correct checks at each stage.

**Acceptance Criteria:**

**Given** the testing strategy from Story 1.3 is documented
**When** the developer runs `pytest`
**Then** the test suite discovers and runs tests from the defined directory structure
**And** Hypothesis is available for property-based test generation
**And** Mutmut is configured for mutation testing on designated modules
**And** test markers distinguish pre-commit tests from PR-time-only tests (e.g., `@pytest.mark.slow`)
**And** at least one passing smoke test exists to validate the framework is operational
**And** pytest configuration in `pyproject.toml` defines default options, markers, and test paths

### Story 1.6: Configure Session Management & Automation

As a developer,
I want Nox configured to orchestrate the full quality pipeline,
So that running `nox` executes linting, type-checking, and testing in one command.

**Acceptance Criteria:**

**Given** Ruff, Mypy, and Pytest are configured from Stories 1.4 and 1.5
**When** the developer runs `nox`
**Then** Nox executes sessions in order: Ruff (lint/format) -> Mypy (type check) -> Pytest (tests)
**And** each session runs in an isolated environment
**And** failure in any session is clearly reported with the failing session identified
**And** `noxfile.py` is committed to the repository root
**And** the developer can run individual sessions (e.g., `nox -s lint`, `nox -s typecheck`, `nox -s tests`)

### Story 1.7: Configure Versioning, Packaging & Documentation

As a developer,
I want Commitizen, check-manifest, edgetest, and Sphinx configured,
So that the project has automated versioning, package integrity checks, dependency management, and documentation generation.

**Acceptance Criteria:**

**Given** the Poetry project structure from Story 1.1 is in place
**When** the developer uses Commitizen for commits
**Then** commit messages follow the conventional commits format and version bumps are automated
**And** `check-manifest` validates that the package manifest includes all necessary files
**And** edgetest is configured for dependency compatibility testing
**And** `sphinx-build` generates HTML documentation from the `docs/` directory using the Furo theme
**And** `sphinx-apidoc` can auto-generate API docs from module docstrings
**And** a Nox session exists for documentation generation (`nox -s docs`)

### Story 1.8: Implement Runtime Logging & Data Assertions Framework

As a developer,
I want a structured logging system with configurable verbosity levels and a data assertions framework,
So that I can diagnose runtime issues efficiently and validate data integrity throughout the pipeline.

**Acceptance Criteria:**

**Given** the project toolchain (Stories 1.4-1.6) is configured
**When** the developer uses the logging and assertions modules
**Then** a structured logging system is available using Python's `logging` module with project-specific configuration
**And** custom verbosity levels are supported (e.g., QUIET, NORMAL, VERBOSE, DEBUG) controllable via CLI flag or environment variable
**And** log output includes timestamps, module names, and configurable formatting
**And** a data assertions module provides helper functions for validating DataFrame shapes, column types, value ranges, and null checks
**And** assertion failures produce clear error messages with the specific validation that failed and the actual vs. expected values
**And** the logging and assertions framework is covered by unit tests
**And** usage examples are documented in the module docstrings

## Epic 2: Data Ingestion & Local Warehouse

User can fetch NCAA data from external sources, persist it locally, and access it with smart caching -- the "Source of Truth" is operational.

### Story 2.1 (Spike): Evaluate Data Sources

As a data scientist,
I want a documented evaluation of available NCAA data sources (Kaggle, KenPom, BartTorvik, ESPN, Nate Silver, etc.),
So that I can make informed decisions about which sources to prioritize based on feasibility, coverage, cost, and rate limits.

**Acceptance Criteria:**

**Given** the project needs external NCAA data to function
**When** the developer reviews the spike findings document
**Then** each candidate source is evaluated for: data coverage (years, stats available), API accessibility (public vs. paid, auth method), rate limits and terms of service, and data format/quality
**And** a recommended priority order of sources is documented with rationale
**And** any licensing or cost implications are clearly noted
**And** the findings are committed as a project document

#### Spike Decisions (Story 2.1) — APPROVED

The data source evaluation (see `specs/research/data-source-evaluation.md`) assessed 18 candidate sources. The following 2 sources are **approved** for MVP implementation in Stories 2.2–2.4:

| # | Source | Access Method | Primary Value | Risk Note |
|:---|:---|:---|:---|:---|
| 1 | **Kaggle MMLM** | `kaggle` CLI/API (free) | Historical game data 1985+, seeds, brackets, MasseyOrdinals (100+ ranking systems) | Low — well-established |
| 2 | **ESPN via cbbpy** | `cbbpy` scraper (free) | Current-season game data, calendar dates, schedule enrichment | Medium — ESPN endpoint instability |

**Story mapping:** Story 2.2 (schema) accommodates fields from all sources. Story 2.3 (connectors) implements one connector per approved source. Story 2.4 (sync CLI) orchestrates all connectors with caching.

**Deferred to post-MVP backlog:** BartTorvik scraping (no Python client — cbbdata is R-only, cbbpy is ESPN-only), Warren Nolan (scrape-only, low priority), sportsdataverse-py (28 open issues, redundant with cbbpy), Nate Silver / SBCB Elo ratings (Substack scraping), KenPom ($20/yr subscription + fragile scraping), EvanMiya (paid), ShotQuality ($3K/yr).

### Story 2.2: Define Internal Data Schema & Repository Layer

As a data scientist,
I want a unified internal data schema (Team, Game, Season entities) with a Repository pattern abstracting storage,
So that all downstream code works against a consistent API regardless of storage backend.

**Acceptance Criteria:**

**Given** the Architecture specifies Team, Game, and Season entities
**When** the developer imports the data layer
**Then** Team, Game, and Season are defined as typed data structures (Pydantic models or dataclasses)
**And** Team includes: `TeamID` (int), `Name` (str), `CanonicalName` (str)
**And** Game includes: `GameID`, `Season`, `Date`, `WTeamID`, `LTeamID`, `WScore`, `LScore`, `Loc`
**And** Season includes: `Year` (int)
**And** a Repository interface abstracts read/write operations (`get_games(season)`, `get_teams()`, `save_games(games)`)
**And** at least one concrete Repository implementation exists (SQLite or Parquet -- decision finalized here)
**And** the repository is covered by unit tests validating round-trip read/write

### Story 2.3: Implement Data Source Connectors

As a data scientist,
I want connectors for each prioritized external data source that fetch raw data and map it to the internal schema,
So that I can ingest NCAA data from multiple sources into a unified format.

**Acceptance Criteria:**

**Given** the spike findings (Story 2.1) identify prioritized sources and the internal schema (Story 2.2) is defined
**When** the developer calls a connector for a specific source
**Then** the connector fetches raw data from the external source
**And** raw data is cleaned and mapped to the internal Team/Game/Season schema
**And** team name normalization maps diverse source-specific names to canonical IDs
**And** each connector handles its source's quirks (authentication, pagination, data format)
**And** connectors raise clear errors on network failures, auth issues, or unexpected data formats
**And** each connector is covered by tests (using mocked API responses)

### Story 2.4: Implement Sync CLI & Smart Caching

As a data scientist,
I want a CLI command `python sync.py --source [kaggle|kenpom|...] --dest <path>` that populates my local store with smart caching,
So that I can fetch historical data once and prefer local data on subsequent runs.

**Acceptance Criteria:**

**Given** data source connectors (Story 2.3) and the Repository layer (Story 2.2) are implemented
**When** the developer runs `python sync.py --source kaggle --dest data/`
**Then** the sync command fetches data from the specified source and persists it via the Repository
**And** `--source all` fetches from all configured sources
**And** on subsequent runs, the caching layer checks for valid local data before making remote API calls
**And** the cache can be bypassed with a `--force-refresh` flag
**And** sync progress is displayed to the user (source being fetched, records written)
**And** the sync command is covered by integration tests validating the full fetch-store-cache cycle

## Epic 3: Exploratory Data Analysis

User can explore ingested data to understand quality, structure, and relationships, producing documented findings that inform feature engineering.

### Story 3.1: Data Quality Audit

As a data scientist,
I want to explore the ingested NCAA data for completeness, consistency, and anomalies,
So that I understand data quality issues before building features or models.

**Acceptance Criteria:**

**Given** the local data store is populated via the Sync CLI (Epic 2)
**When** the data scientist runs the data quality audit notebook
**Then** the notebook documents the schema and structure of all ingested tables (row counts, column types, date ranges)
**And** missing values are quantified per column and per season
**And** duplicate records are identified and documented
**And** anomalies and edge cases are flagged (e.g., 2020 COVID year with no tournament, unusual scores, neutral-site games)
**And** data quality issues are summarized with recommended cleaning actions
**And** the notebook is committed to the repository with reproducible outputs

### Story 3.2: Statistical Exploration & Relationship Analysis

As a data scientist,
I want to explore statistical distributions, correlations, and patterns in the NCAA data,
So that I can identify signals and relationships worth pursuing in feature engineering.

**Acceptance Criteria:**

**Given** the data quality audit (Story 3.1) has identified the usable dataset
**When** the data scientist runs the exploration notebook
**Then** scoring distributions are visualized (home vs. away, by seed, by conference, over time)
**And** home/away/neutral venue effects are quantified
**And** correlations between available statistics and tournament outcomes are analyzed
**And** strength-of-schedule and conference-strength signals are explored
**And** seed vs. actual performance patterns are documented (upset rates by seed matchup)
**And** all visualizations use Plotly for interactive inline rendering
**And** the notebook is committed to the repository with reproducible outputs

### Story 3.3: Document Findings & Feature Engineering Recommendations

As a data scientist,
I want a synthesized document of EDA findings with actionable recommendations,
So that Epic 4 (Feature Engineering) has clear direction on what features to build and what data issues to address.

**Acceptance Criteria:**

**Given** the data quality audit (Story 3.1) and statistical exploration (Story 3.2) are complete
**When** the data scientist reads the findings document
**Then** confirmed data quality issues are listed with specific cleaning recommendations
**And** promising feature engineering approaches are identified with supporting evidence from EDA
**And** signals worth pursuing are ranked by expected predictive value
**And** known limitations and caveats in the data are documented
**And** the document is committed as a project reference for Epic 4 planning

## Epic 4: Feature Engineering Suite

User can transform raw game data into ML-ready features including sequential stats, opponent adjustments, graph centrality, and canonical team IDs, with chronological data serving that enforces temporal boundaries.

### Story 4.1 (Spike): Research Feature Engineering Techniques

As a data scientist,
I want a documented survey of feature engineering techniques used in sports prediction (especially NCAA tournament contexts),
So that I can make informed decisions about which transformations to implement based on proven approaches and EDA findings.

**Acceptance Criteria:**

**Given** EDA findings (Epic 3) have identified promising signals and the project needs feature engineering direction
**When** the data scientist reviews the spike findings document
**Then** opponent adjustment methods are documented (e.g., ridge regression efficiency, SRS-style solvers)
**And** sequential/momentum feature approaches are catalogued (rolling averages, streaks, recency weighting)
**And** graph-based features are surveyed (PageRank, betweenness centrality, clustering coefficient)
**And** Kaggle March Machine Learning Mania discussion boards are reviewed for community-proven techniques
**And** each technique is assessed for feasibility, complexity, and expected predictive value
**And** a prioritized implementation plan is documented
**And** the findings are committed as a project document

### Story 4.2: Implement Chronological Data Serving API

As a data scientist,
I want a `get_chronological_season(year)` API that streams game data in strict date order with temporal boundary enforcement,
So that I can train models with walk-forward validation without risk of data leakage.

**Acceptance Criteria:**

**Given** the Repository layer (Epic 2) contains populated game data
**When** the developer calls `get_chronological_season(2023)`
**Then** games are returned strictly ordered by date within the season
**And** the API makes it impossible to access games beyond a specified cutoff date
**And** requesting data for a future date raises a clear error
**And** the 2020 COVID year returns regular season data but flags the absence of tournament games
**And** the API supports iteration (streaming) for memory-efficient processing of large seasons
**And** temporal boundary enforcement is covered by unit tests including edge cases (season boundaries, same-day games)

### Story 4.3: Implement Canonical Team ID Mapping & Data Cleaning

As a data scientist,
I want a normalization layer that maps diverse team names to canonical IDs, integrates supplementary lookup tables, and ingests Massey Ordinal rankings,
So that features are computed on consistent, clean data and all pre-computed multi-system ratings are available to the feature pipeline with temporal fidelity.

**Acceptance Criteria:**

**Given** ingested data may contain varying team name formats across sources
**When** the developer runs the normalization pipeline
**Then** all team name variants are mapped to a single canonical TeamID per team using `MTeamSpellings.csv`
**And** the mapping handles common variations (abbreviations, mascots, "State" vs "St.", etc.)
**And** unmapped team names raise warnings with suggested matches
**And** the cleaning pipeline is idempotent (running it twice produces the same result)

**And** `MNCAATourneySeeds.csv` is parsed into structured fields: `seed_num` (integer 1–16), `region` (W/X/Y/Z), `is_play_in` (bool — True for seeds with 'a'/'b' suffix)
**And** `MTeamConferences.csv` provides a `(season, team_id) → conference` lookup for every season available

**And** `MMasseyOrdinals.csv` is ingested with all 100+ ranking systems, preserving the `RankingDayNum` temporal field for each record
**And** a **coverage gate** verifies whether SAG (Sagarin) and WLK (Whitlock) are present for all 23 seasons (2003–2025): if either has gaps, the fallback composite is MOR+POM+DOL (all confirmed full-coverage, all margin-based)
**And** the following composite building blocks are available (modeler selects at feature-serving time):
  - **Option A:** Simple average of selected systems' ordinal ranks (e.g., `(SAG + POM + MOR + WLK) / 4` if coverage confirmed; fallback `(MOR + POM + DOL) / 3`)
  - **Option B:** Weighted ensemble with system weights derived from prior-season CV log loss
  - **Option C:** PCA reduction of all available systems to N principal components (capturing ≥90% variance)
  - **Option D:** Pre-tournament snapshot — use only ordinals from the last available `RankingDayNum ≤ 128` per system per season
**And** ordinal feature normalization options are provided: rank delta between teams (primary matchup feature), percentile (bounded [0,1]), and z-score per season
**And** the pre-computed Colley ("COL") and Massey ("MAS") systems from `MMasseyOrdinals.csv` are available as alternatives to reimplementing those solvers in Story 4.6
**And** the normalization and ingestion module is covered by unit tests with known name-variant fixtures and known ordinal coverage assertions

### Story 4.4: Implement Sequential Transformations

As a data scientist,
I want rolling windows, EWMA, momentum, streak, per-possession, and Four Factor features computed from chronologically ordered game data,
So that I can capture recent team form, efficiency, and trends as predictive features without data leakage.

**Acceptance Criteria:**

**Given** chronological game data is available via the serving API (Story 4.2)
**When** the developer applies sequential transformations to a team's game history
**Then** rolling averages are computed over configurable windows of 5, 10, and 20 games (plus full-season aggregate) for all EDA Tier 1 stats; all three window sizes are parallel feature columns — not competing features, but modeler-configurable parameters of the same building block
**And** all sequential features respect chronological ordering (no future data leakage)
**And** features are computed using vectorized operations (numpy/pandas) per NFR1

**And** EWMA (Exponentially Weighted Moving Average) is implemented with configurable α (range 0.10–0.30; recommended start α=0.15–0.20 mapping to effective window of 9–12 games); uses `pandas.DataFrame.ewm(alpha=α).mean()` per team per season
**And** a momentum/trajectory feature is produced as `ewma_fast − ewma_slow` (rate of change of efficiency; positive = improving form into tournament)

**And** win/loss streaks are encoded as a signed integer: `+N` for winning streak of N games, `−N` for losing streak, capturing pure win/loss sequence dynamics independent of efficiency magnitude

**And** per-possession normalization is applied to all counting stats: `possessions = FGA − OR + TO + 0.44 × FTA`; stat values are divided by possession count to remove pace confound
**And** Four Factors are computed: `eFG% = (FGM + 0.5 × FGM3) / FGA`, `ORB% = OR / (OR + opponent_DR)`, `FTR = FTA / FGA`, `TO% = TO / possessions`

**And** home court encoding converts `loc` to a numeric feature: H=+1, A=−1, N=0 (or one-hot for tree-based models); EDA-confirmed +2.2pt home margin advantage
**And** time-decay game weighting applies the BartTorvik formula before rolling aggregations: games >40 days old lose 1% weight per day, with a floor of 60% (`weight = max(0.6, 1 − 0.01 × max(0, days_ago − 40))`)
**And** `rescale_overtime(score, num_ot)` from Story 4.2 is applied to raw scores before any aggregation (normalizes OT games to 40-minute equivalent)

**And** edge cases are handled: season start with insufficient history, mid-season breaks
**And** sequential transformations are covered by unit tests validating correctness and temporal integrity

### Story 4.5: Implement Graph Builders & Centrality Features

As a data scientist,
I want to convert season schedules into NetworkX directed graphs and compute PageRank, betweenness centrality, HITS (hub + authority), and clustering coefficient features,
So that I can quantify transitive team strength, structural schedule position, and schedule diversity as predictive features.

**Acceptance Criteria:**

**Given** game data for a season is available
**When** the developer builds a season graph and computes centrality features
**Then** the season schedule is converted to a NetworkX directed graph with edges directed winner←loser (loser "votes for" winner quality), using `nx.from_pandas_edgelist()` — no iterrows
**And** edge weights are margin-of-victory capped at 25 points (`min(margin, 25)`) to prevent extreme-blowout distortion
**And** optional date-recency weighting multiplies edge weight by a recency factor (e.g., games in the last 20 days get 1.5× weight)

**And** **PageRank** is computed (directed, margin-weighted, `nx.pagerank(G, alpha=0.85, weight="weight")`) — captures transitive win-chain strength (2 hops vs. SoS 1 hop); peer-reviewed NCAA validation: 71.6% vs. 64.2% naive win-ratio (Matthews et al. 2021)
**And** **Betweenness centrality** is computed (`nx.betweenness_centrality()`) — captures structural "bridge" position; distinct signal from both strength (PageRank) and schedule quality (SoS)
**And** **HITS** hub and authority scores are both computed via a single `nx.hits()` call; authority score is exposed (largely redundant with PageRank, r≈0.908, but zero additional cost); hub score is a distinct signal ("quality schedule despite losses")
**And** **Clustering coefficient** is computed (`nx.clustering()`) — schedule diversity metric: low clustering = broad cross-conference scheduling

**And** walk-forward incremental update strategy is implemented: PageRank uses power-iteration warm start (initialize with previous solution; 2–5 iterations instead of 30–50); betweenness is fully recomputed each time step (O(V×E) per step; pre-computed and stored by game date for walk-forward use over 40+ seasons)

**And** graph features can be computed incrementally as games are added (for walk-forward use in Story 4.7)
**And** graph builders are covered by unit tests with known small-graph fixtures including PageRank convergence and betweenness structural correctness assertions

### Story 4.6: Implement Batch Opponent Adjustment Rating Systems

As a data scientist,
I want batch linear algebra rating solvers (SRS, Ridge, Colley) that produce opponent-adjusted team ratings for the full season,
So that I can generate features that account for schedule strength and quality of competition.

**Acceptance Criteria:**

**Given** full-season game data with scores and team matchup information is available
**When** the developer runs the opponent adjustment solvers
**Then** **SRS (Simple Rating System)** is implemented as the Group A canonical representative: fixed-point iteration solve (`r_i(k+1) = avg_margin_i + avg(r_j for all opponents j)`); convergence guaranteed for connected schedules (~3,000–5,000 iterations); produces margin-adjusted batch rating
**And** **Ridge regression** is implemented as the Group A λ-parameterized variant: regularized SRS via `sklearn.linear_model.Ridge`; λ configurable in range 10–100 (default λ=20 for full-season data); exposes shrinkage as a modeler-visible tuning knob without providing a distinct signal from SRS
**And** **Colley Matrix** is implemented as the Group B representative (win/loss only): Cholesky solve for the Colley matrix `C[i,i] = 2 + t_i`, `C[i,j] = -n_ij`; or the pre-computed "COL" system from `MMasseyOrdinals.csv` (ingested in Story 4.3) is used as an alternative — implementation choice resolved during Story 4.6 development

**And** all three solvers produce full-season pre-tournament snapshots (ratings as of the last regular-season game), not in-season incremental updates (that is Story 4.8's responsibility)
**And** the solvers handle edge cases: teams with very few games, structurally isolated conference subgraphs (near-singular sub-blocks), unconnected schedule components
**And** outputs are validated against the pre-computed "MAS" (Massey) system in `MMasseyOrdinals.csv` for sanity-check benchmarking

**And** note: Elo (dynamic game-by-game rating as a feature building block) is implemented in Story 4.8, not here — that story covers the stateful/incremental rating approach
**And** the solvers are covered by unit tests including convergence assertions (SRS), lambda-sensitivity tests (Ridge), and win/loss isolation tests (Colley)

### Story 4.7: Implement Stateful Feature Serving

As a data scientist,
I want a feature serving layer that combines all active feature transformations into a temporally-safe feature matrix, with in-fold probability calibration and matchup-level feature support,
So that models receive a consistent, leakage-free feature matrix with calibrated probability outputs.

**Acceptance Criteria:**

**Given** sequential, graph, batch rating, dynamic rating, and normalization features are implemented (Stories 4.3–4.6, 4.8)
**When** the developer requests features for a model training run
**Then** the serving layer combines all active feature transformations into a unified feature matrix via declarative configuration (specify which building blocks to activate)
**And** features are served in strict chronological order matching the data serving API (Story 4.2)
**And** the serving layer enforces that no feature computation uses future data relative to the prediction point
**And** the serving layer supports both stateful (per-game iteration) and stateless (batch) consumption modes

**And** **Massey ordinal temporal slicing** is enforced: for each game at date D, only ordinals with `RankingDayNum` published ≤ D are used — prevents ordinal leakage during walk-forward backtesting
**And** **matchup-level features** are computed as team_A − team_B deltas: seed differential (`seed_num_A − seed_num_B`), ordinal rank deltas, Elo delta, SRS delta — these are the primary matchup signals for tournament prediction

**And** **probability calibration** is applied in-fold (not post-hoc): isotonic regression or cubic-spline calibration fitted on training fold predictions, applied to test fold predictions; the `goto_conversion` Python package is assessed as an alternative calibration implementation
**And** calibration is always in-fold to prevent leakage — fitting calibration on held-out data is NOT acceptable

**And** `gender_scope` and `dataset_scope` are configurable parameters on the feature server (e.g., men's vs. women's data; Kaggle-only vs. ESPN-enriched games)
**And** the feature serving pipeline is covered by integration tests validating end-to-end temporal integrity, calibration leakage prevention, and matchup-level delta correctness

### Story 4.8: Implement Dynamic Rating Features (Elo Feature Building Block)

As a data scientist,
I want a game-by-game Elo rating system that produces team ratings as features for the walk-forward feature pipeline,
So that I can capture in-season trajectory and momentum in addition to the full-season batch ratings from Story 4.6.

**Note:** This story implements Elo ratings as a **feature building block** (a rating computed from game history to feed as input to another model, e.g., XGBoost). Story 5.3 implements Elo as a complete predictive **model** — these are architecturally distinct.

**Acceptance Criteria:**

**Given** chronological game data is available via the serving API (Story 4.2)
**When** the developer runs the Elo feature generator on a season's game history
**Then** Elo ratings are updated game-by-game from a configurable initial rating (default 1500): `r_new = r_old + K_eff × (actual − expected)`, where `expected = 1 / (1 + 10^((r_opponent − r_team)/400))`
**And** the K-factor is configurable and supports variable-K: K=56 (early season) → K=38 (regular season) → K=47.5 (tournament games)
**And** margin-of-victory scaling is supported: `K_eff = K × min(margin, max_margin)^0.85` (Silver/SBCB formula; diminishing returns on blowouts); `max_margin` is configurable
**And** home-court adjustment subtracts a configurable number of Elo points (default 3–4) from the home team's effective rating before computing expected outcome
**And** season mean-reversion is applied between seasons: regress a configurable fraction (default 25%, range 20–35%) of each team's rating toward its conference mean to account for roster turnover
**And** a pre-tournament Elo snapshot (as of the last regular-season game) is available as a team-level feature column compatible with Story 4.7 matchup delta computation
**And** Elo updates are walk-forward compatible: computed incrementally game-by-game from the chronological serving API with no future data leakage
**And** the Elo feature generator is covered by unit tests validating rating updates, margin scaling, home court adjustment, and season mean-reversion correctness

---

## Epic 5: Core Modeling Framework

User can train, predict, and persist models using a standardized contract that supports both stateful (Elo) and stateless (XGBoost) approaches.

### Story 5.1 (Spike): Research Modeling Approaches

As a data scientist,
I want a documented survey of modeling approaches used for NCAA tournament prediction,
So that I can ensure the Model ABC supports all viable approaches and select the best reference implementations.

**Acceptance Criteria:**

**Given** the project needs to support diverse modeling approaches
**When** the data scientist reviews the spike findings document
**Then** Kaggle March Machine Learning Mania discussion boards are reviewed across multiple competition years
**And** stateful model approaches are catalogued (Elo variants, Glicko, TrueSkill, custom rating systems)
**And** stateless model approaches are catalogued (XGBoost, logistic regression, neural nets, ensemble methods)
**And** hybrid approaches are documented (e.g., Elo features fed into XGBoost)
**And** requirements for the Model ABC are derived from the survey (what interface must support all approaches)
**And** reference models to implement first are recommended with rationale
**And** the findings are committed as a project document

### Story 5.2: Define Model ABC & Plugin Registry

As a data scientist,
I want an abstract base class (`Model`) with a plugin-registry architecture,
So that I can implement custom models that plug into the training and evaluation pipeline without modifying core code.

**Acceptance Criteria:**

**Given** the spike findings (Story 5.1) define the interface requirements
**When** the developer creates a new model by subclassing `Model`
**Then** the `Model` ABC enforces implementation of `fit(X, y)`, `predict_proba(X)`, `save(path)`, `load(path)`, and `get_config()` abstract methods
**And** `fit(X: pd.DataFrame, y: pd.Series) -> None` is the unified training interface for all model types (sklearn naming convention)
**And** `predict_proba(X: pd.DataFrame) -> pd.Series` returns calibrated P(team_a wins) in [0.0, 1.0] for each row in X (unified for stateful and stateless)
**And** `load(cls, path: Path) -> Self` is a classmethod returning `Self` (PEP 673 / Python 3.12) so that `EloModel.load(path)` is typed as `EloModel`, not `Model`
**And** `get_config() -> ModelConfig` returns the Pydantic-validated config; `ModelConfig` (Pydantic BaseModel) is the base class for all model configs
**And** a `StatefulModel(Model)` subclass is defined with: (1) concrete template `fit()` that reconstructs `Game` objects from X and calls `update()` per game; (2) concrete template `predict_proba()` that dispatches to `_predict_one()` per row; (3) abstract hooks `update(game: Game)`, `_predict_one(team_a_id, team_b_id)`, `start_season(season)`, `get_state()`, `set_state(state)`
**And** stateless models (XGBoost, logistic regression) implement `Model` directly — NO separate `StatelessModel` subclass exists
**And** the plugin registry provides `@register_model("name")` decorator, `get_model(name) -> type[Model]`, and `list_models() -> list[str]`; built-in models auto-register on package import; external users register via `@register_model` before invoking the pipeline
**And** a minimal logistic regression implementation (`LogisticRegressionModel(Model)`) is included as a test fixture (not as a production reference model) — demonstrates the stateless `Model` contract in ~30 lines
**And** the ABC and registry are covered by unit tests including the logistic regression test fixture
**And** type annotations satisfy `mypy --strict`

**Design Reference:** `specs/research/modeling-approaches.md` Section 5 (complete interface pseudocode, import-verified across 3 code review rounds)

### Story 5.3: Implement Reference Stateful Model (Elo)

As a data scientist,
I want a working Elo rating system as the reference stateful model,
So that I have a proven baseline for tournament prediction and a template for building other stateful models.

**Acceptance Criteria:**

**Given** the Model ABC (Story 5.2) is defined and `EloFeatureEngine` (Story 4.8, `transform.elo`) is available
**When** the developer trains the Elo model on historical game data
**Then** `EloModel(StatefulModel)` wraps `EloFeatureEngine` from `transform.elo` — it does NOT re-implement Elo from scratch; `fit(X, y)` is inherited from `StatefulModel` (calls `update()` per reconstructed game)
**And** `update(game: Game)` delegates to `EloFeatureEngine.update_game()` to advance ratings
**And** `start_season(season: int)` delegates to `EloFeatureEngine.start_new_season(season)` for mean reversion
**And** `_predict_one(team_a_id: int, team_b_id: int) -> float` returns P(team_a wins) via the Elo expected-score formula using current ratings; public prediction is via inherited `predict_proba(X: pd.DataFrame) -> pd.Series`
**And** `EloModelConfig(ModelConfig)` is the Pydantic config with parameters: `initial_rating`, `k_early`, `early_game_threshold`, `k_regular`, `k_tournament`, `margin_exponent`, `max_margin`, `home_advantage_elo`, `mean_reversion_fraction` — defaults matching `EloConfig` from Story 4.8 (see `specs/research/modeling-approaches.md` §5.5 and §6.4)
**And** `get_state() -> dict[str, Any]` returns the ratings dict; `set_state(state)` restores it
**And** `save(path: Path)` JSON-dumps ratings dict + config; `load(cls, path: Path) -> Self` reconstructs from JSON
**And** the model registers via the plugin registry as `"elo"`
**And** the Elo model is validated against known rating calculations on a small fixture dataset
**And** the model is covered by unit tests for rating updates, `_predict_one`, state persistence (`get_state`/`set_state`), and `save`/`load` round-trip

**Design Reference:** `specs/research/modeling-approaches.md` §6.1 (implementation approach), §5.5 (EloModelConfig), §6.4 (hyperparameter ranges)

### Story 5.4: Implement Reference Stateless Model (XGBoost)

As a data scientist,
I want an XGBoost wrapper as the reference stateless model,
So that I have a powerful gradient-boosting baseline and a template for building other batch-trained models.

**Acceptance Criteria:**

**Given** the Model ABC (Story 5.2) is defined and `StatefulFeatureServer` (Epic 4, Story 4.7) provides feature matrices
**When** the developer trains the XGBoost model on a feature matrix
**Then** `XGBoostModel(Model)` wraps `xgboost.XGBClassifier` implementing `Model` directly (no `StatefulModel` subclass — stateless models bypass the per-game lifecycle)
**And** `fit(X: pd.DataFrame, y: pd.Series)` calls `XGBClassifier.fit(X, y, eval_set=..., early_stopping_rounds=...)` using the validation split from `X`
**And** `predict_proba(X: pd.DataFrame) -> pd.Series` returns `XGBClassifier.predict_proba(X)[:, 1]` — P(team_a wins) as calibrated probabilities (XGBoost `binary:logistic` objective)
**And** `XGBoostModelConfig(ModelConfig)` is the Pydantic config with: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `early_stopping_rounds` — see `specs/research/modeling-approaches.md` §5.5 and §6.4 for defaults and tuning ranges
**And** label balance is verified before training: if `StatefulFeatureServer` assigns team_a/team_b non-randomly (e.g., always winner = team_a), `scale_pos_weight` must be set accordingly; document the convention in the implementation
**And** `save(path: Path)` calls `clf.save_model(str(path / "model.ubj"))` (XGBoost UBJSON native format, stable across versions) and writes config JSON to `path / "config.json"`
**And** `load(cls, path: Path) -> Self` instantiates `XGBClassifier()` then calls `clf.load_model(str(path / "model.ubj"))` — `load_model` is an instance method, NOT a class method
**And** the model registers via the plugin registry as `"xgboost"`
**And** the model is covered by unit tests validating `fit`/`predict_proba`/`save`/`load` round-trip

**Design Reference:** `specs/research/modeling-approaches.md` §6.2 (implementation approach), §5.5 (XGBoostModelConfig), §6.4 (hyperparameter ranges), §5.7 (persistence format)

### Story 5.5: Implement Model Run Tracking & Training CLI

As a data scientist,
I want model run metadata tracked and a CLI for launching training jobs,
So that I can reproduce results, compare runs, and train models from the terminal.

**Acceptance Criteria:**

**Given** the Model ABC and reference models (Stories 5.2-5.4) are implemented
**When** the developer runs `python -m ncaa_eval.cli train --model elo --start-year 2015 --end-year 2025`
**Then** a ModelRun record is created with: RunID, ModelType, Hyperparameters (JSON), Timestamp, and GitHash
**And** Prediction records are created for each game prediction with: RunID, GameID, PredWinProb
**And** ModelRun and Prediction records are persisted to the local store
**And** training progress is displayed via progress bars in the terminal
**And** results summary (metrics, run metadata) is printed on completion
**And** the CLI supports `--model` flag accepting any registered plugin model name (built-in: `"elo"`, `"xgboost"`; external user-registered names also work)
**And** the CLI and tracking are covered by integration tests validating the full train-track-persist cycle

**Note:** `fit(X, y)` is the canonical training entry point for all models (see Story 5.2). The CLI's `train` sub-command constructs the feature matrix via `StatefulFeatureServer` and calls `model.fit(X, y)`.

## Epic 6: Evaluation & Validation Engine

User can evaluate models with probabilistic metrics, calibration analysis, walk-forward cross-validation, and Monte Carlo tournament simulation.

### Story 6.1: Implement Metric Library (scikit-learn + numpy)

As a data scientist,
I want a metric library computing Log Loss, Brier Score, ROC-AUC, ECE, and reliability diagram data,
So that I can evaluate model quality across multiple dimensions using vectorized operations.

**Acceptance Criteria:**

**Given** a set of model predictions (probabilities) and actual outcomes
**When** the developer calls the metric functions
**Then** Log Loss is computed via `sklearn.metrics.log_loss`
**And** Brier Score is computed via `sklearn.metrics.brier_score_loss`
**And** ROC-AUC is computed via `sklearn.metrics.roc_auc_score`
**And** ECE (Expected Calibration Error) is computed using numpy vectorized operations (not available in scikit-learn)
**And** reliability diagram bin data is generated using `sklearn.calibration.calibration_curve` with numpy for additional binning statistics
**And** all metric functions accept numpy arrays and return scalar or array results
**And** no Python `for` loops are used in metric calculations (vectorization enforced per NFR1)
**And** each metric function is covered by unit tests with known expected values
**And** edge cases are handled (perfect predictions, all-same-class, single prediction)

### Story 6.2: Implement Walk-Forward Cross-Validation Splitter

As a data scientist,
I want a "Leave-One-Tournament-Out" cross-validation splitter with strict temporal boundaries,
So that I can backtest models across multiple years without data leakage.

**Acceptance Criteria:**

**Given** historical game data spanning multiple seasons
**When** the developer uses the CV splitter to generate train/test folds
**Then** each fold uses one tournament year as the test set and all prior years as training data
**And** strict temporal boundaries ensure no future data appears in any training fold
**And** the 2020 COVID year is handled gracefully: models receive training data but no test evaluation is attempted
**And** the splitter yields `(train_data, test_data, year)` tuples for each fold
**And** the splitter is compatible with both stateful models (chronological iteration) and stateless models (batch splits)
**And** the splitter is covered by unit tests validating temporal integrity and 2020 handling
**And** fold boundaries are deterministic and reproducible

### Story 6.3: Implement Parallel Cross-Validation Execution

As a data scientist,
I want cross-validation folds and model evaluations to run in parallel via joblib,
So that multi-year backtests complete faster by utilizing all available CPU cores.

**Acceptance Criteria:**

**Given** the CV splitter (Story 6.2) generates multiple folds
**When** the developer runs a parallelized backtest
**Then** independent CV folds execute concurrently using `joblib.Parallel`
**And** the number of parallel workers is configurable (default: all cores)
**And** progress is reported during parallel execution (fold completion, elapsed time)
**And** results from all folds are collected and aggregated into a summary DataFrame
**And** the 10-year Elo backtest (training & inference) completes in under 60 seconds per the PRD performance target
**And** parallel execution produces identical results to sequential execution (determinism)
**And** parallel CV is covered by integration tests comparing parallel vs. sequential results

### Story 6.4 (Spike): Research Tournament Simulation Confidence

As a data scientist,
I want a documented analysis of how to improve confidence in tournament simulation predictions given limited historical data,
So that I can make informed decisions about simulation methodology and result interpretation.

**Acceptance Criteria:**

**Given** the tournament only happens once per year, limiting the historical dataset
**When** the data scientist reviews the spike findings document
**Then** statistical approaches for improving simulation confidence are evaluated (bootstrapping, Bayesian methods, ensemble simulations)
**And** the impact of sample size on simulation stability is quantified (how many simulations are needed for stable Expected Points)
**And** methods for computing confidence intervals on simulation outputs are documented
**And** recommendations for the simulation implementation (Story 6.5) are provided
**And** the findings are committed as a project document

### Story 6.5: Implement Monte Carlo Tournament Simulator

As a data scientist,
I want a simulation engine that generates N bracket realizations from a model's probability matrix,
So that I can compute Expected Points and Bracket Distribution metrics for tournament strategy.

**Acceptance Criteria:**

**Given** a model's pairwise win probability matrix for tournament teams
**When** the developer runs `simulate_tournament(probs, n=10000)`
**Then** N complete bracket realizations are generated by sampling game outcomes from the probability matrix
**And** the number of simulations N is configurable (default 10,000)
**And** each simulation respects the tournament bracket structure (64-team single elimination, post-First Four — play-in games are excluded)
**And** results include: per-team advancement frequencies by round, most likely bracket (max likelihood), and bracket distribution statistics
**And** simulation leverages numpy vectorization for batch sampling (not Python loops per game)
**And** simulation progress is reported for long runs
**And** the simulator is covered by unit tests validating bracket structure integrity and statistical properties (e.g., probabilities sum to 1 per matchup)

### Story 6.6: Implement Tournament Scoring with User-Defined Point Schedules

As a data scientist,
I want to apply configurable point schedules to simulated or actual brackets,
So that I can evaluate model value under different pool scoring rules and optimize my entry strategy.

**Acceptance Criteria:**

**Given** simulated brackets (Story 6.5) or actual tournament results
**When** the developer applies a scoring schedule to bracket results
**Then** built-in scoring schedules are available: Standard (1-2-4-8-16-32), Fibonacci (1-1-2-3-5-8), and Seed-Difference Bonus
**And** custom scoring schedules can be defined via configuration (dict or callable)
**And** "Expected Points" is computed by averaging scores across all N simulated brackets
**And** "Bracket Distribution" shows the score distribution across simulations (percentiles, histogram data)
**And** scoring integrates with the plugin registry for user-defined custom scoring functions
**And** scoring is covered by unit tests with known bracket fixtures and expected point totals

## Epic 7: Lab & Presentation Dashboard

User can visualize model performance via interactive Streamlit dashboards including leaderboards, reliability diagrams, bracket visualizer, and pool ROI simulations.

### Story 7.1: Build Plotly Adapters for Jupyter Lab Visualization

As a data scientist,
I want API methods on model and evaluation objects that return interactive Plotly figures,
So that I can visualize calibration, metrics, and results directly in Jupyter notebooks.

**Acceptance Criteria:**

**Given** a trained model with evaluation results available
**When** the developer calls visualization methods (e.g., `model.plot_calibration()`, `eval.plot_metrics()`)
**Then** each method returns a `plotly.graph_objects.Figure` object that renders inline in Jupyter
**And** reliability diagrams show predicted vs. actual probability with bin counts
**And** metric comparison charts support multi-model overlay
**And** all figures use the project's functional color palette (Green/Red/Neutral)
**And** figures are interactive (hover tooltips, zoom, pan)
**And** evaluation metrics and logs are also available as Pandas DataFrames for ad-hoc analysis
**And** real-time progress bars are provided for long-running training loops and evaluations when executed in Jupyter cells (e.g., via `tqdm.notebook` or `tqdm.auto`)
**And** adapters are covered by unit tests validating figure object structure and data content

### Story 7.2: Build Streamlit App Shell & Navigation

As a data scientist,
I want a Streamlit multipage app with sidebar navigation, dark mode, and persistent global filters,
So that I can seamlessly switch between Lab and Presentation views while maintaining context.

**Acceptance Criteria:**

**Given** the dashboard application is launched via `poetry run streamlit run dashboard/app.py`
**When** the user opens the application in a browser
**Then** the app renders in Dark Mode by default with Wide Mode layout
**And** a persistent sidebar provides navigation between "Lab" (Research Mode) and "Presentation" (Entry Mode) sections
**And** global filters for Tournament Year, Model Version, and Scoring Format are available in the sidebar
**And** filter selections persist across page navigation via `st.session_state`
**And** `@st.cache_data` is used for loading heavy datasets (model results, game data) to ensure sub-500ms interaction response
**And** the dashboard imports and calls `ncaa_eval` functions exclusively (no direct file IO)
**And** monospace fonts are applied to all data tables per the UX spec

### Story 7.3: Build Lab Page -- Backtest Leaderboard

As a data scientist,
I want a sortable leaderboard comparing all trained models by various metrics,
So that I can quickly identify the best-performing models and spot trends.

**Acceptance Criteria:**

**Given** model run results are persisted in the local store (Epic 5)
**When** the user navigates to the Lab Leaderboard page
**Then** all model runs are displayed in a sortable table with columns for each metric (LogLoss, Brier, ROC-AUC, ECE)
**And** the table supports sorting by any metric column
**And** `st.metric` diagnostic cards display top-line KPIs with performance deltas vs. baseline
**And** conditional formatting (Green-to-Red gradients) highlights model outliers per the UX spec
**And** clicking a model run ID navigates to the Model Deep Dive view (Story 7.4)
**And** the leaderboard filters by the global Tournament Year and Model Version selections
**And** data loads within the 500ms interaction response target via `@st.cache_data`

### Story 7.4: Build Lab Page -- Model Deep Dive & Reliability Diagrams

As a data scientist,
I want detailed diagnostic views for a specific model showing calibration, confusion, and feature importance,
So that I can understand where a model succeeds and fails beyond aggregate metrics.

**Acceptance Criteria:**

**Given** the user has selected a specific model run from the Leaderboard (Story 7.3)
**When** the user views the Model Deep Dive page
**Then** a reliability diagram (predicted vs. actual probability) is rendered via `st.plotly_chart`
**And** the diagram clearly identifies model over-confidence or under-confidence per the PRD success metric
**And** a metric explorer allows drill-down by year, round, seed matchup, or conference
**And** feature importance is displayed (for stateless models like XGBoost)
**And** all visualizations use the functional color palette and are interactive (Plotly)
**And** breadcrumb navigation shows context (e.g., Home > Lab > v1.2-GraphModel)

### Story 7.5: Build Presentation Page -- Bracket Visualizer

As a data scientist,
I want an interactive tournament bracket visualization with clickable matchup details and Game Theory sliders,
So that I can visually inspect specific predictions and explore "what-if" scenarios.

**Acceptance Criteria:**

**Given** a model's probability matrix and simulated bracket results are available
**When** the user navigates to the Bracket Visualizer page
**Then** a 64-team single-elimination bracket tree is rendered (post-First Four — play-in games are excluded) using a custom Streamlit component (technology determined by Story 7.7 spike)
**And** the bracket requires Wide Mode and displays all four regions simultaneously without horizontal scrolling
**And** clicking a game node opens a detail panel showing matchup features (efficiency stats, graph centrality, head-to-head)
**And** Game Theory sliders in the sidebar (Upset Aggression, Chalk Bias, Seed-Weight) perturb the model's base probabilities in real-time using the mechanism defined in Story 7.7 spike
**And** slider adjustments update the bracket visualization without altering the underlying model data
**And** the user can flag a specific bracket configuration as a "Candidate Entry"

### Story 7.7 (Spike): Research Game Theory Slider Mechanism

As a data scientist,
I want a documented analysis of how Game Theory sliders (Upset Aggression, Chalk Bias, Seed-Weight) should mathematically transform a model's base win probabilities,
So that the Bracket Visualizer (Story 7.5) can implement real-time probability perturbation with a sound mathematical foundation.

**Acceptance Criteria:**

**Given** the UX spec defines sliders that "perturb the model's base probabilities" without specifying the mechanism
**When** the data scientist reviews the spike findings document
**Then** candidate mathematical transformations are evaluated (e.g., logit-space additive adjustments, multiplicative scaling, Bayesian prior blending)
**And** each approach is assessed for: intuitive user behavior (slider up = more upsets), numerical stability (probabilities remain valid 0-1), and reversibility (slider at neutral = original probabilities)
**And** the recommended approach is documented with formula, examples, and edge case analysis
**And** slider parameter ranges and default values are specified
**And** the findings are committed as a project document

### Story 7.6: Build Presentation Page -- Pool Scorer & Point Outcome Analysis

As a data scientist,
I want to configure pool-specific scoring rules and analyze the range of possible point outcomes,
So that I can understand my bracket's scoring potential under different pool formats.

**Acceptance Criteria:**

**Given** a model's probability matrix and the tournament simulator (Epic 6) are available
**When** the user navigates to the Pool Scorer page
**Then** the user can input pool scoring rules (Standard, Fibonacci, Seed-Difference Bonus, or custom)
**And** clicking "Analyze Outcomes" runs the Monte Carlo simulator with the selected scoring rules to produce a distribution of possible point totals
**And** simulation progress is displayed via `st.progress` to prevent UI freezing during 10k+ iterations
**And** results display the point outcome distribution (min, max, median, percentiles) and a histogram of simulated scores
**And** the user can click "Generate Submission" to export the Final Entry as CSV/JSON formatted for the target pool
**And** simulation results are cached to avoid re-running on page navigation

### Story 7.8: Write Comprehensive User Guide

As a data scientist,
I want a comprehensive guide explaining the evaluation metrics, model types, and how to interpret the results,
So that I can understand what the platform measures and make informed decisions based on its outputs.

**Acceptance Criteria:**

**Given** the core platform (Epics 1-6) and dashboard (Epic 7) are functional
**When** the user reads the user guide
**Then** all evaluation metrics are explained (Log Loss, Brier Score, ROC-AUC, ECE) with intuitive descriptions and examples
**And** model types are documented (Stateful vs. Stateless) with guidance on when to use each
**And** result interpretation is covered: how to read reliability diagrams, what calibration means, and how to use bracket simulations
**And** the tournament scoring systems are explained (Standard, Fibonacci, Seed-Difference Bonus)
**And** the guide is written in Sphinx-compatible RST or Markdown and integrated into the auto-generated documentation
**And** the guide is accessible from the project's documentation site

### Story 7.9: Create Step-by-Step Tutorials

As a data scientist,
I want step-by-step tutorials for common tasks,
So that I can quickly learn how to use the platform's key workflows.

**Acceptance Criteria:**

**Given** the platform is functional and the user guide (Story 7.8) is available
**When** the user follows a tutorial
**Then** a "Getting Started" tutorial covers the full pipeline: sync data, train a model, evaluate, and view results in the dashboard
**And** a "How to Create a Custom Model" tutorial walks through subclassing the Model ABC, registering via the plugin registry, and running evaluation
**And** a "How to Add a Custom Metric" tutorial demonstrates extending the evaluation engine via the plugin registry
**And** each tutorial includes runnable code examples and expected outputs
**And** tutorials are written in Sphinx-compatible RST or Markdown and integrated into the auto-generated documentation

## Post-MVP Backlog

Items identified during development for future consideration. These are not scheduled for any sprint but may be promoted into epics/stories later.

### Model ABC Plugins — LightGBM (Story 5.1 spike decision, 2026-02-23)

`LightGBMModel(Model)` — stateless Model ABC plugin wrapping `lightgbm.LGBMClassifier`. Near-identical pattern to `XGBoostModel` (~50 lines); same GBDT family but with leaf-wise tree growth and native categorical support. 2025 winner tested LightGBM and found XGBoost superior on NCAA data; deferred because XGBoost already covers the GBDT equivalence group.

- **Effort:** ~50 lines — `fit(X, y)` / `predict_proba(X)` / `save(path)` / `load(path)` wrapping `LGBMClassifier`
- **Distinctness:** Low — same GBDT family as XGBoost; minimal additional signal on small NCAA datasets
- **Source:** Story 5.1 spike — `specs/research/modeling-approaches.md` §3.2, §7.1 (Group A equivalence)
- **Template:** Follow `XGBoostModel` pattern exactly; `@register_model("lightgbm")`

### Model ABC Plugins — CatBoost (Story 5.1 spike decision, 2026-02-23)

`CatBoostModel(Model)` — stateless Model ABC plugin wrapping `catboost.CatBoostClassifier`. Ordered boosting with native categorical handling. 2025 winner tested CatBoost and found it underperformed XGBoost; deferred for same reason as LightGBM.

- **Effort:** ~50 lines — same pattern as `XGBoostModel`
- **Distinctness:** Low — same GBDT family; ordered boosting provides marginal benefit on NCAA-sized data
- **Source:** Story 5.1 spike — `specs/research/modeling-approaches.md` §3.3, §7.1 (Group A equivalence)
- **Template:** Follow `XGBoostModel` pattern; `@register_model("catboost")`

### Model ABC Plugins — Glicko-2 & TrueSkill (Story 5.1 spike decision, 2026-02-23)

`Glicko2Model(StatefulModel)` and `TrueSkillModel(StatefulModel)` — uncertainty-quantified rating models. Each adds rating deviation / skill variance beyond Elo, but both converge toward standard Elo for full-season snapshots (30+ games per team reduces uncertainty gap). Deferred: marginal signal, weak competition validation (occasional top-25, not top-10).

- **Effort:** ~150 lines each — implement `update(game)`, `_predict_one()`, `start_season()`, `get_state()`, `save()`/`load()` using `glicko2` or `trueskill` PyPI packages
- **Distinctness:** Low-Medium — RD/volatility are genuine new parameters but converge for full-season data
- **Source:** Story 5.1 spike — `specs/research/modeling-approaches.md` §2.2, §2.3, §2.5
- **Template:** Follow `EloModel` (Story 5.3) as the stateful reference; `@register_model("glicko2")` / `@register_model("trueskill")`

### Model ABC Plugins — LSTM & Transformer (Story 5.1 spike decision, 2026-02-23)

`LSTMModel(Model)` and `TransformerModel(Model)` — deep learning models for sequential NCAA tournament prediction. arXiv:2508.02725 (Habib 2025) reports Transformer-BCE achieves highest AUC (0.8473) but poor calibration; LSTM-Brier achieves best calibration. Deferred: no competition wins, small data disadvantage vs. GBDT, high implementation complexity (PyTorch/TensorFlow training loops).

- **Effort:** High — custom PyTorch/TF training loop, architecture design, sequence data formatting
- **Distinctness:** Moderate — captures temporal game sequences not in tabular features; but GBDT still outperforms on small NCAA data (Grinsztajn et al. 2022, NeurIPS)
- **Source:** Story 5.1 spike — `specs/research/modeling-approaches.md` §3.5
- **Note:** These require PyTorch or TensorFlow as new dependencies — add only as optional extras in pyproject.toml

### Model ABC Plugins — Bayesian Logistic Regression (Story 5.1 spike decision, 2026-02-23)

`BayesianLogisticRegressionModel(Model)` — Bayesian LR with informative priors (via `pymc` or `bambi`). Won MMLM 2015 (Bradshaw) and 2017 (Landgraf). Deferred: standard `LogisticRegressionModel` is the Story 5.2 test fixture and covers the linear model equivalence group; Bayesian variant adds uncertainty quantification but higher implementation complexity.

- **Effort:** Medium — `pymc` or `bambi` dependency; MCMC sampling is slower than sklearn LR
- **Distinctness:** Slight extension of logistic regression — posterior uncertainty is useful but adds complexity
- **Source:** Story 5.1 spike — `specs/research/modeling-approaches.md` §3.4, §7.1 (Group B equivalence)

### LRMC (Logistic Regression Markov Chain) Rating System

Models tournament outcomes as a Markov chain where each team's win probability against any opponent is derived from game-by-game outcomes via logistic regression. Results in a steady-state probability distribution over tournament outcomes. Documented in Edwards 2021 (top Kaggle MMLM solution writeup) as one of several rating systems computed from scratch.

- **Complexity:** High — requires implementing a Markov chain transition matrix; more complex than SRS/Ridge/Colley batch solvers
- **Distinctness:** Distinct from SRS/Ridge (batch least-squares) and Elo (dynamic updates); provides Markov-chain-derived win probabilities rather than point-differential-based ratings
- **Source:** Story 4.1 spike — `specs/research/feature-engineering-techniques.md` Section 6.4 (Edwards 2021) and Section 7.2 (Distinct Building Blocks table, Story 4.6)
- **Deferred because:** High implementation complexity relative to marginal distinctness; no peer-reviewed NCAA-specific validation; Edwards 2021 used it but ranked mid-pack; LRMC may not provide independent signal beyond SRS + Elo combination

### TrueSkill / Glicko-2 Rating Systems

Uncertainty-quantified Elo variants that explicitly model rating variance (uncertainty) per team. TrueSkill uses a factor graph with Gaussian belief propagation; Glicko-2 uses RD (Rating Deviation) and volatility parameters. Both are available as Python packages (`trueskill`, `glicko2`).

- **Complexity:** Medium-High — requires understanding factor graphs (TrueSkill) or RD update formulas (Glicko-2)
- **Distinctness:** Distinct from Elo in that they quantify rating uncertainty per team — a team with high uncertainty (few games or inconsistent results) has lower effective rating certainty. Marginal signal over Elo for pre-tournament snapshots with 30+ games per team.
- **Source:** Story 4.1 spike — `specs/research/feature-engineering-techniques.md` Section 7.2 (Distinct Building Blocks, "TrueSkill / Glicko-2", Story 4.6) and Section 6.8 (Community Techniques table)
- **Deferred because:** Marginal information gain over Elo at full-season (30+ games per team reduces uncertainty gap); occasional top-25 community validation but not consistently top-10; implementation cost not justified until Elo (Story 4.8) is validated

### Nate Silver / SBCB Elo Rating Scraping

Scrape Nate Silver's Silver Bulletin (Substack) posts for free Elo ratings. Silver publishes pre-tournament Elo rankings (~350 D1 teams, history back to 1950) that could serve as an additional feature source or model benchmark. His enhanced Elo system includes margin-of-victory diminishing returns, per-team home court advantage, and variable K-factor — worth replicating or comparing against.

- **Access:** Substack HTML scraping (no API, no structured data export)
- **Cost:** Free tier includes Elo tables; paid tier ($8/mo) for full SBCB/COOPER model outputs
- **Risk:** Substack layout changes could break scraper; Silver may move to COOPER platform in 2026
- **Source:** Story 2.1 spike — `specs/research/data-source-evaluation.md`, Section 9

### BartTorvik Direct Scraping

Scrape barttorvik.com for adjusted efficiency metrics (AdjOE, AdjDE), T-Rank ratings, and Four Factors data (2008+). The `cbbdata` REST API is R-only with no Python client, and `cbbpy` does not provide BartTorvik data (ESPN scraper only). Direct scraping of barttorvik.com or use of the [andrewsundberg Kaggle dataset](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset) for historical T-Rank CSVs are the viable Python access paths.

- **Access:** HTML scraping of barttorvik.com (no official API) or Kaggle CSV dataset
- **Cost:** Free
- **Risk:** HTML scraping is fragile; site layout changes could break scraper. Kaggle dataset may lag behind current season.
- **Value:** Adjusted efficiency metrics are the gold standard for team strength estimation. Kaggle MasseyOrdinals provides partial coverage via "POM" system (KenPom-derived), but direct BartTorvik metrics (especially Four Factors and recency-weighted ratings) would be more granular.
- **Source:** Story 2.3 scoping — confirmed `cbbdata` is R-only, `cbbpy` is ESPN-only. See `specs/research/data-source-evaluation.md`, Section 2.

### Warren Nolan Scraping

Scrape warrennolan.com for NET rankings, RPI, and Nitty Gritty strength-of-schedule reports. Provides official NCAA evaluation metrics used by the selection committee.

- **Access:** HTML scraping (no API, no structured data export)
- **Cost:** Free
- **Risk:** HTML scraping is fragile; categorized as "Deferred Scrape-Only" in Story 2.1 research document. Inclusion in the original Spike Decisions MVP table contradicted the research recommendation.
- **Value:** NET rankings are the NCAA's official team evaluation metric (replaced RPI in 2018). Useful for tournament selection committee modeling but not essential for game outcome prediction.
- **Source:** Story 2.1 spike — `specs/research/data-source-evaluation.md`, Section 4. Story 2.3 scoping deferred due to contradiction with research doc classification.
