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

#### Spike Decisions (Story 2.1)

The data source evaluation (see `specs/research/data-source-evaluation.md`) assessed 18 candidate sources. The following 4 sources are selected for MVP implementation in Stories 2.2–2.4:

| # | Source | Access Method | Primary Value |
|:---|:---|:---|:---|
| 1 | **Kaggle MMLM** | `kaggle` CLI/API (free) | Historical game data 1985+, seeds, brackets, MasseyOrdinals (100+ ranking systems) |
| 2 | **BartTorvik / cbbdata API** | REST API (free) | Adjusted efficiency metrics (AdjOE/AdjDE), T-Rank, Four Factors 2008+ |
| 3 | **sportsdataverse-py** | Python package (free) | ESPN API wrapper, play-by-play data 2002+, schedules |
| 4 | **Warren Nolan** | HTML scraping (free) | NET rankings, RPI, Nitty Gritty strength-of-schedule reports |

**Story mapping:** Story 2.2 (schema) must accommodate fields from all 4 sources. Story 2.3 (connectors) implements one connector per source. Story 2.4 (sync CLI) orchestrates all connectors with caching.

**Deferred to post-MVP backlog:** Nate Silver / SBCB Elo ratings (Substack scraping), KenPom ($20/yr subscription + fragile scraping), EvanMiya (paid), ShotQuality ($3K/yr).

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
I want a normalization layer that maps diverse team names to canonical IDs and applies standard data cleaning,
So that features are computed on consistent, clean data regardless of the original source.

**Acceptance Criteria:**

**Given** ingested data may contain varying team name formats across sources
**When** the developer runs the normalization pipeline
**Then** all team name variants are mapped to a single canonical TeamID per team
**And** the mapping handles common variations (abbreviations, mascots, "State" vs "St.", etc.)
**And** unmapped team names raise warnings with suggested matches
**And** data cleaning applies scaling, categorical encoding, and normalization as needed
**And** the cleaning pipeline is idempotent (running it twice produces the same result)
**And** the normalization module is covered by unit tests with known name-variant fixtures

### Story 4.4: Implement Sequential Transformations

As a data scientist,
I want rolling averages, streaks, and momentum features computed from chronologically ordered game data,
So that I can capture recent team form and trends as predictive features.

**Acceptance Criteria:**

**Given** chronological game data is available via the serving API (Story 4.2)
**When** the developer applies sequential transformations to a team's game history
**Then** rolling averages are computed over configurable windows (e.g., last 5, 10, 20 games)
**And** win/loss streaks are tracked with current streak length and direction
**And** momentum features capture "last N games" performance trends
**And** all sequential features respect chronological ordering (no future data leakage)
**And** features are computed using vectorized operations (numpy/pandas) per NFR1
**And** edge cases are handled (season start with insufficient history, mid-season breaks)
**And** sequential transformations are covered by unit tests validating correctness and temporal integrity

### Story 4.5: Implement Graph Builders & Centrality Features

As a data scientist,
I want to convert season schedules into NetworkX graph objects and compute centrality metrics,
So that I can quantify strength of schedule and network position as predictive features.

**Acceptance Criteria:**

**Given** game data for a season is available
**When** the developer builds a season graph and computes centrality features
**Then** the season schedule is converted to a NetworkX directed graph (teams as nodes, games as edges)
**And** edge weights reflect margin of victory or win probability
**And** PageRank is computed to quantify overall team strength within the schedule network
**And** betweenness centrality and other relevant metrics are available as features
**And** graph features can be computed incrementally as games are added (for walk-forward use)
**And** graph builders are covered by unit tests with known small-graph fixtures

### Story 4.6: Implement Opponent Adjustments

As a data scientist,
I want linear algebra solvers for opponent-adjusted efficiency statistics,
So that I can generate features that account for strength of competition.

**Acceptance Criteria:**

**Given** game data with scores and team matchup information is available
**When** the developer runs the opponent adjustment solver
**Then** offensive and defensive efficiency ratings are computed adjusted for opponent strength
**And** the solver uses linear algebra methods (e.g., ridge regression, least squares) per FR5
**And** ratings can be computed incrementally for walk-forward compatibility
**And** the solver handles edge cases (teams with very few games, unconnected schedule components)
**And** opponent-adjusted stats are validated against known benchmarks (e.g., KenPom-style ratings on historical data)
**And** the solver is covered by unit tests including convergence and edge case tests

### Story 4.7: Implement Stateful Feature Serving

As a data scientist,
I want a feature serving layer that feeds chronologically-ordered, combined features to models during training,
So that models receive a consistent, temporally-safe feature matrix regardless of which transformations are active.

**Acceptance Criteria:**

**Given** sequential, graph, opponent adjustment, and normalization features are implemented (Stories 4.3-4.6)
**When** the developer requests features for a model training run
**Then** the serving layer combines all active feature transformations into a unified feature matrix
**And** features are served in strict chronological order matching the data serving API (Story 4.2)
**And** the serving layer enforces that no feature computation uses future data relative to the prediction point
**And** feature configuration is declarative (specify which transformations to include)
**And** the serving layer supports both stateful (per-game iteration) and stateless (batch) consumption modes
**And** the feature serving pipeline is covered by integration tests validating end-to-end temporal integrity

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
**Then** the ABC enforces implementation of `train`, `predict`, and `save` methods
**And** the ABC distinguishes between stateful models (per-game state updates) and stateless models (batch training)
**And** stateful models support `update(game)` for incremental state updates and `get_state()` / `set_state()` for persistence
**And** the plugin registry allows registering custom models by name (e.g., `@register_model("my_elo")`)
**And** registered models are discoverable at runtime (e.g., `get_model("my_elo")`)
**And** the ABC and registry are covered by unit tests including a minimal mock model implementation
**And** type annotations satisfy `mypy --strict`

### Story 5.3: Implement Reference Stateful Model (Elo)

As a data scientist,
I want a working Elo rating system as the reference stateful model,
So that I have a proven baseline for tournament prediction and a template for building other stateful models.

**Acceptance Criteria:**

**Given** the Model ABC (Story 5.2) is defined and the chronological serving API (Epic 4, Story 4.2) is available
**When** the developer trains the Elo model on historical game data
**Then** the Elo model consumes games via `get_chronological_season(year)` from the chronological serving API, updating team ratings after each game
**And** the K-factor and initial rating are configurable hyperparameters
**And** home-court advantage adjustment is supported
**And** season-to-season state persistence is implemented (ratings carry forward with optional regression to mean)
**And** `predict(team_a, team_b)` returns a win probability derived from the rating difference
**And** the model registers via the plugin registry as `"elo"`
**And** the Elo model is validated against known rating calculations on a small fixture dataset
**And** the model is covered by unit tests for rating updates, prediction, and state persistence

### Story 5.4: Implement Reference Stateless Model (XGBoost)

As a data scientist,
I want an XGBoost wrapper as the reference stateless model,
So that I have a powerful gradient-boosting baseline and a template for building other batch-trained models.

**Acceptance Criteria:**

**Given** the Model ABC (Story 5.2) is defined and the feature serving layer (Epic 4) provides feature matrices
**When** the developer trains the XGBoost model on a feature matrix
**Then** the model wraps `xgboost.XGBClassifier` with the `Model` ABC interface
**And** `train(X, y)` fits the classifier on the provided feature matrix and labels
**And** `predict(X)` returns calibrated win probabilities (not raw scores)
**And** hyperparameters are configurable via constructor or config dict
**And** `save()` persists the trained model to disk and `load()` restores it
**And** the model registers via the plugin registry as `"xgboost"`
**And** the model is covered by unit tests validating train/predict/save/load round-trip

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
**And** the CLI supports `--model` flag accepting any registered plugin model name
**And** the CLI and tracking are covered by integration tests validating the full train-track-persist cycle

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

### Nate Silver / SBCB Elo Rating Scraping

Scrape Nate Silver's Silver Bulletin (Substack) posts for free Elo ratings. Silver publishes pre-tournament Elo rankings (~350 D1 teams, history back to 1950) that could serve as an additional feature source or model benchmark. His enhanced Elo system includes margin-of-victory diminishing returns, per-team home court advantage, and variable K-factor — worth replicating or comparing against.

- **Access:** Substack HTML scraping (no API, no structured data export)
- **Cost:** Free tier includes Elo tables; paid tier ($8/mo) for full SBCB/COOPER model outputs
- **Risk:** Substack layout changes could break scraper; Silver may move to COOPER platform in 2026
- **Source:** Story 2.1 spike — `specs/research/data-source-evaluation.md`, Section 9
