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
| NFR3 | Epic 5 (Partial) | Plugin-registry extensibility — model and scoring registries implemented; metric and feature-generator registries deferred to Post-MVP |
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

### Epic 8: Codebase Improvements & Technical Debt Resolution
All Category 3 (obviously-need-fixing) findings from the multi-agent codebase audit are resolved; PO direction gathered on Category 1 & 2 items.
**Source:** `_bmad-output/planning-artifacts/codebase-audit-report.md` and addenda; `_bmad-output/planning-artifacts/epic-8-codebase-improvements.md`

### Epic 9: Audit-Driven Enhancements
New features and fixes approved by the PO in the Epic 8 decision log — product gaps, usability improvements, and the `feature_config`-as-model-concern refactor that enables ensemble modeling.

### Epic 10: Ensemble Modeling Framework
Users can define a stacked ensemble of any base models, train the full stack end-to-end in one call, and generate live bracket predictions via a game-aware meta-learner. Depends on Epic 9 (Story 9.2).

### Epic X: Cookiecutter Project Template
Extract NCAA_eval's project structure, toolchain, BMAD workflow configuration, and conventions into a reusable cookiecutter template for future Python ML projects.
**Timing:** Post-project-completion — not scheduled for active development sprints.

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
I want Commitizen, check-manifest, and Sphinx configured,
So that the project has automated versioning, package integrity checks, dependency management, and documentation generation.

**Acceptance Criteria:**

**Given** the Poetry project structure from Story 1.1 is in place
**When** the developer uses Commitizen for commits
**Then** commit messages follow the conventional commits format and version bumps are automated
**And** `check-manifest` validates that the package manifest includes all necessary files
**And** ~~edgetest is configured for dependency compatibility testing~~ (Deferred: removed in Story 8.11 — never automated in CI)
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

### Story 1.9: Restructure docs/ as Pure Sphinx Source Directory

As a developer,
I want `docs/` to be a pure Sphinx source directory with planning specs moved to a top-level `specs/` directory,
So that all documentation in `docs/` is processed by Sphinx and the directory has a single, clear purpose.

**Acceptance Criteria:**

**Given** Sphinx is configured in `docs/` (Story 1.7)
**When** `nox -s docs` is run
**Then** STYLE_GUIDE.md, TESTING_STRATEGY.md, and all testing/ guides are rendered as HTML pages in the Sphinx output alongside the API reference
**And** `docs/` contains only Sphinx source files (no excluded planning artifacts)
**And** planning specs live at `specs/` (project root) with `specs/archive/` for legacy documents
**And** the Sphinx HTML navigation has three sections: Developer Guides, Testing Guides, and API Reference
**And** `check-manifest` passes cleanly with updated ignore patterns
**And** `nox` (full pipeline: lint, typecheck, tests) passes with no regressions

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
**And** all visualizations use matplotlib for static PNG rendering (Plotly inline outputs caused ~800 MB notebook files — see Story 3.1 findings)
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
**And** the `{contents}` TOC directive is removed from `docs/user-guide.md` (conflicts with Furo's built-in right-sidebar TOC). Make sure to search other documentation and see if other TOCs need removal.
**And** the project `README.md` is reviewed and enhanced. At the very least it should be updated to include a link to the GitHub Pages documentation site (`https://dhilgart.github.io/NCAA_eval/`), but there should also be though given to what else should be added and what should be removed. Also pay attention to what status bars at the top should be added.

## Epic 8: Codebase Improvements & Technical Debt Resolution

All Category 3 (obviously-need-fixing) findings from the multi-agent codebase audit are resolved; PO direction gathered on Category 1 & 2 items. Full story ACs are in [`_bmad-output/planning-artifacts/epic-8-codebase-improvements.md`](_bmad-output/planning-artifacts/epic-8-codebase-improvements.md).

**Source:** `_bmad-output/planning-artifacts/codebase-audit-report.md` and addenda; `_bmad-output/planning-artifacts/epic-8-codebase-improvements.md`

### Story 8.1: Code Architecture Cleanup — Simulation Module Split & Kitchen Sink Refactors

Split `simulation.py` (1,291 lines, 7+ responsibilities) and `dashboard/lib/filters.py` (621 lines, kitchen-sink) into focused modules. Decompose `run_training()` God Function.

**Priority:** High (Category 3) | **Full ACs:** `epic-8-codebase-improvements.md` §8.1

### Story 8.2: Expose Public APIs & Eliminate Private Attribute Access

Add public methods to `EloFeatureEngine`, create `Calibrator` Protocol/ABC, fix cross-module private attribute access, and use typed scoring registry. Eliminate all `_`-prefixed cross-boundary accesses.

**Priority:** High (Category 3) | **Full ACs:** `epic-8-codebase-improvements.md` §8.2

### Story 8.3: Fix Data Pipeline Resilience — ESPN Error Handling, Retry Logic, Typer Decoupling

Add retry logic for ESPN API calls, improve error handling, decouple ESPN connector from Typer CLI context, and fix quiet mode behavior.

**Priority:** High (Category 3) | **Full ACs:** `epic-8-codebase-improvements.md` §8.3

### Story 8.4: Fix Docstring Style Violations & Documentation Gaps

Correct all docstrings to Google style (Args/Returns sections), add missing docstrings to public functions/methods, and fix the `__init__` docstring anti-pattern.

**Priority:** High (Category 3) | **Full ACs:** `epic-8-codebase-improvements.md` §8.4

### Story 8.5: Testing Gaps — Missing Tests & Dead Code Cleanup

Add missing unit tests for uncovered public functions, remove dead code, and fix test isolation issues.

**Priority:** High (Category 3) | **Full ACs:** `epic-8-codebase-improvements.md` §8.5

### Story 8.6: Type Safety & Configuration Improvements

Eliminate remaining `Any` annotations, add proper return types to all functions, fix `Optional[X]` → `X | None` patterns, and address other mypy findings.

**Priority:** High (Category 3) | **Full ACs:** `epic-8-codebase-improvements.md` §8.6

### Story 8.7: Sprint Housekeeping & CI/CD Improvements

Update sprint status and implementation artifacts, fix CI/CD pipeline issues, and clean up stale configuration.

**Priority:** Medium (Category 3) | **Full ACs:** `epic-8-codebase-improvements.md` §8.7

### Story 8.8: Dashboard UX Quick Fixes

Fix cosmetic UX issues in the Streamlit dashboard: label corrections, layout improvements, and display consistency fixes.

**Priority:** Medium (Category 3) | **Full ACs:** `epic-8-codebase-improvements.md` §8.8

### Story 8.9: Add PEP 20, SOLID & Pure Function Gates to PR Template + Codebase PEP 20 Review

Add engineering quality gates (PEP 20, SOLID, pure function checklist) to the PR review template, and audit the existing codebase against these principles.

**Priority:** Medium (Category 3) | **Full ACs:** `epic-8-codebase-improvements.md` §8.9

### Story 8.10: Documentation Command E2E Integration Tests

Add end-to-end integration tests that execute the documentation build pipeline (`sphinx-build`, `nox -s docs`) to prevent regressions.

**Priority:** Medium (Category 3) | **Full ACs:** `epic-8-codebase-improvements.md` §8.10

### Story 8.11: Fix Testing Documentation Staleness & Marker Gaps

Update testing docs to match actual test organization, add missing pytest markers, and ensure `pytest.ini`/`pyproject.toml` marker declarations are complete.

**Priority:** Medium (Category 3) | **Full ACs:** `epic-8-codebase-improvements.md` §8.11

### Story 8.12: Epics & Backlog Grooming — Track All Deferred Items

Groom the epics and Post-MVP Backlog to ensure all deferred items from prior spikes and audit findings are tracked. Update sprint-status.yaml and backlog accordingly.

**Priority:** Medium (Category 3) | **Full ACs:** `epic-8-codebase-improvements.md` §8.12

### Story 8.13: Gather PO Direction on Category 1 & 2 Items

Walk the PO through all Category 1 (judgment call) and Category 2 (nice-to-have) audit findings and record their decisions in `po-decision-log-epic8.md`. Decisions drive the Epic 9 story list.

**Priority:** High (prerequisite for Epic 9) | **Full ACs:** `epic-8-codebase-improvements.md` §8.13

---

## Epic 9: Audit-Driven Enhancements

Focused improvements identified by the Epic 8 codebase audit and approved by the PO in the decision log (`po-decision-log-epic8.md`). These are low-to-medium effort items that address product gaps, usability, and documentation accuracy.

### Story 9.1: Kaggle Submission Export

As a **data scientist**,
I want to **export my model's predictions in Kaggle March Machine Learning Mania submission format**,
So that **I can submit my bracket predictions directly to the Kaggle competition**.

**Acceptance Criteria:**

**Given** a trained model's probability matrix is available
**When** the user clicks "Export Kaggle Submission" in the dashboard (or runs a CLI command)
**Then** a CSV file is generated with columns `ID` and `Pred` for all 2,278 possible team matchups
**And** the `ID` column uses the Kaggle format `YYYY_TeamID1_TeamID2` (lower ID first)
**And** the `Pred` column contains the model's win probability for TeamID1
**And** the file conforms to the Kaggle `SampleSubmission.csv` schema

**Source:** Audit item 1.3; PRD §1 (competitive submission workflow)

### Story 9.2: Feature Config as Model-Level Concern

As a **data scientist**,
I want to **embed feature engineering configuration directly in my model class**,
So that **my model always receives inputs in the correct format, I can experiment with different feature combinations by passing constructor kwargs, and loaded model artifacts carry their own feature requirements without external configuration files**.

**Acceptance Criteria:**

**Given** any concrete `Model` subclass (`XGBoostModel`, `LogisticRegressionModel`, `EloModel`)
**When** the developer instantiates the model
**Then** the model exposes a `feature_config: FeatureConfig` attribute derived from its constructor kwargs
**And** feature-relevant kwargs (e.g., `batch_rating_types`, `graph_features_enabled`, `ordinal_composite`) are accepted at `__init__` and threaded into the model's `FeatureConfig`
**And** `run_training()` reads `model.feature_config` to build the feature server instead of using the hardcoded defaults in `_setup_feature_server()`
**And** `model.save(path)` persists a `feature_config.json` sidecar alongside model weights
**And** `model.load(path)` reads that sidecar and reconstructs the `FeatureConfig` so a loaded model knows exactly what columns it expects
**And** after `fit()`, every stateless model stores `self.feature_names_: list[str]` — the ordered list of feature columns it was trained on
**And** `FeatureConfig.calibration_method` is removed from `FeatureConfig` and added to `ModelConfig` (calibration is a model-output concern, not a feature-computation concern)
**And** `EloModel` uses a minimal `FeatureConfig` (no batch ratings, no ordinals, `elo_enabled=True`) since it reconstructs `Game` objects from metadata columns only
**And** existing CLI behavior (`ncaa-eval train`) is unchanged — the CLI instantiates model classes whose constructors carry default feature configs

**Source:** Audit item 1.6; `src/ncaa_eval/cli/train.py:90-100`; design spec `specs/ensemble-architecture.md` §2
**Prerequisite for:** Epic 10 (all stories)

### Story 9.3: Feature Importance for Elo and Logistic Regression

As a **data scientist**,
I want to **see feature importance / interpretability information for all model types, not just XGBoost**,
So that **I can understand what drives predictions across Elo, Logistic Regression, and XGBoost models**.

**Acceptance Criteria:**

**Given** a trained model is selected in the Model Deep Dive dashboard page
**When** the user views the Feature Importance section
**Then** XGBoost shows feature importance (existing behavior, unchanged)
**And** Logistic Regression shows coefficient values as feature importance
**And** Elo shows team rating values and/or rating-based metrics as interpretability information
**And** the "not available for stateful models" message is replaced with meaningful Elo interpretability

**Source:** Audit item 1.15; `dashboard/pages/3_Model_Deep_Dive.py`

### Story 9.4: Fix Public API Documentation

As a **developer**,
I want to **have accurate documentation of import paths for the ncaa_eval package**,
So that **the Style Guide matches reality and I know how to import public symbols**.

**Acceptance Criteria:**

**Given** the Style Guide claims `from ncaa_eval import EloModel` should work
**When** the developer reads the Style Guide
**Then** documented import paths match actual importable paths
**And** the Style Guide is updated to document the actual submodule import paths (e.g., `from ncaa_eval.model.elo import EloModel`)

**Source:** Audit item 2.18; `src/ncaa_eval/__init__.py:1-3`, `docs/STYLE_GUIDE.md`

### Story 9.5: Post-Sync Data Validation

As a **data scientist**,
I want to **have automatic validation checks run after data sync completes**,
So that **I can detect data quality issues (missing games, duplicates, team reference errors) before they silently corrupt downstream predictions**.

**Acceptance Criteria:**

**Given** a data sync (`ncaa-eval sync` or `python sync.py`) completes
**When** the sync finishes downloading and persisting data
**Then** a validation step runs automatically checking:
  - Game count per season is within expected range (±10% of historical average)
  - No duplicate games exist (same teams, same day)
  - All team IDs in games reference valid entries in the teams table
**And** validation results are logged at INFO level with a summary
**And** validation warnings do not block the sync (non-fatal) but are clearly visible

**Source:** Audit item 2.20; PRD §4.4

### Story 9.6: Revisit Skipped Audit Decisions

As a **product owner**,
I want to **review and make decisions on the Epic 8 audit items that were deferred during Story 8.13**,
So that **no potential improvements are permanently lost and I can choose which ones to promote into implementation stories**.

**Acceptance Criteria:**

**Given** the `po-decision-log-epic8.md` file with items marked `S — skip, come back later`
**When** the PO reviews each skipped item
**Then** each of the following audit items receives a final decision (Implement, Defer to Post-MVP, or Accept as-is):
  - 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10 (Category 2 items skipped in batch)
  - 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.21 (additional Category 2 items)
  - P2-5, P2-6 (Pass 2 addendum items)
  - P3-20 (Pass 3 addendum item)
**And** any items decided as "Implement" are converted into new Epic 9 stories (added to this epic or scheduled for the next available sprint)
**And** any items decided as "Defer to Post-MVP" are added to the Post-MVP Backlog in `epics.md`
**And** any items decided as "Accept as-is" are marked resolved in `po-decision-log-epic8.md`
**And** `po-decision-log-epic8.md` has no remaining items in "S — skip" status after this story completes

**Source:** Story 8.13 session — items deferred 2026-03-09; `po-decision-log-epic8.md`

### Story 9.7: Game Theory Slider Implementation

As a **data scientist**,
I want to **adjust Upset Aggression, Chalk Bias, and Seed-Weight sliders in the dashboard sidebar to perturb the model's base probabilities in real time**,
So that **I can explore bracket outcomes under different risk strategies without retraining the model**.

**Acceptance Criteria:**

**Given** the dashboard Presentation page
**When** the user moves an Upset Aggression, Chalk Bias, or Seed-Weight slider
**Then** the model's base win probabilities are perturbed using the formulas established in Story 7.7 spike research
**And** the bracket visualization updates in real time to reflect the perturbed probabilities
**And** the slider controls are documented in the user guide with a clear explanation of each slider's effect
**And** the "NOT YET IMPLEMENTED" banner added in Story 8.4 is removed

**Source:** Audit item 1.1; Story 7.7 spike (`specs/research/`); PO decision 2026-03-09

### Story 9.8: User-Editable Bracket

As a **data scientist**,
I want to **click matchups in the bracket view to override the model's predicted winner**,
So that **I can score my own picks against historical results and evaluate the model's guidance relative to my own judgment**.

**Acceptance Criteria:**

**Given** the Bracket Visualizer dashboard page
**When** the user clicks a matchup to override the predicted winner
**Then** the bracket downstream of that matchup updates to reflect the user's pick
**And** the Pool Scorer scores the user-edited bracket (not just the model's most-likely bracket)
**And** user overrides persist for the session and can be reset to model predictions with a "Reset" button

**Architecture note:** Requires a `UserOverrideProvider` that wraps an existing `ProbabilityProvider` and substitutes user picks at specific bracket nodes, as identified by the Architect in audit item 1.2.

**Source:** Audit item 1.2; PO decision 2026-03-09

### Story 9.9: CLI `predict` Command

As a **data scientist**,
I want to **run `ncaa-eval predict <run-id>` from the command line to generate win-probability predictions for current-season matchups**,
So that **I can get predictions without launching the dashboard or running a notebook**.

**Acceptance Criteria:**

**Given** a saved model run (identified by `<run-id>`)
**When** the user runs `ncaa-eval predict <run-id>`
**Then** the model is loaded from the run artifact directory
**And** win probabilities are computed for all current-season games (or a specified date range)
**And** output is written to stdout as CSV or optionally to a file via `--output`
**And** the command is documented in the CLI reference (`docs/`)

**Source:** Audit item 1.11; PO decision 2026-03-09

### Story 9.10: Custom Metric Plugin Registry

As a **data scientist**,
I want to **register a custom metric function and have it appear in the metric explorer and leaderboard alongside the built-in metrics**,
So that **I can evaluate models on domain-specific criteria without modifying library source code**.

**Acceptance Criteria:**

**Given** a function decorated with `@register_metric("my_metric")`
**When** the user runs a backtest or opens the metric explorer dashboard
**Then** `my_metric` appears alongside `log_loss`, `brier_score`, and the other built-in metrics
**And** the `@register_metric` decorator and `MetricRegistry` are publicly exported from `ncaa_eval`
**And** Story 7.9 tutorial "How to Add a Custom Metric" is updated to accurately document the registry API (replacing the stub that described a non-existent feature)
**And** the feature-generator registry is NOT implemented in this story (remains in Post-MVP Backlog)

**Source:** Audit item P3-17; PO decision 2026-03-09 (Custom — implement metric registry only)

## Epic 10: Ensemble Modeling Framework

Users can define a stacked ensemble of any base models, train the full stack end-to-end in one call, and generate live bracket predictions — without manually managing out-of-fold alignment, feature server coordination, or meta-learner input construction.

**Design spec:** `specs/ensemble-architecture.md`
**Prerequisite:** Story 9.2 (Feature Config as Model-Level Concern) must be complete before any Epic 10 story begins.

### Story 10.1: StackedEnsemble Class and OOF Training Pipeline

As a **data scientist**,
I want to **define a stacked ensemble by listing base models and a meta-learner and train the whole thing in one `run_training()` call**,
So that **I can build an ensemble that learns optimal, game-context-dependent weights without manually orchestrating out-of-fold prediction generation or alignment**.

**Acceptance Criteria:**

**Given** a `StackedEnsemble` instance with `base_models`, `meta_learner`, and `contextual_features`
**When** the user calls `run_training(ensemble, data_dir=..., start_year=..., end_year=..., output_dir=..., model_name=...)`
**Then** `run_training()` detects the `StackedEnsemble` type and routes to `_run_ensemble_training()`
**And** for each base model, a walk-forward backtest is run using that model's own `feature_config` to produce out-of-fold (OOF) predictions
**And** OOF predictions from all base models are aligned by `game_id` via inner join; a warning is logged if >5% of games are dropped by the join
**And** a meta-training DataFrame is assembled with columns `[pred_base_0, pred_base_1, ..., <contextual_features>]`
**And** the meta-learner is trained on the meta-training DataFrame
**And** each base model is retrained on the full dataset (all seasons)
**And** the ensemble artifact is saved: each base model, the meta-learner, and a manifest recording base model names, `contextual_features`, and the run IDs of the OOF backtest runs
**And** `StackedEnsemble.feature_config` returns the union of all base models' `feature_config`s (used internally by the ensemble's own `predict_proba()`)

**Source:** `specs/ensemble-architecture.md` §3–§4
**Prerequisite:** Story 9.2

### Story 10.2: Ensemble Inference Interface

As a **data scientist**,
I want to **generate bracket predictions and evaluate an ensemble on historical data using the same interfaces I use for single models**,
So that **ensembles compose transparently with the existing evaluation and bracket-generation infrastructure**.

**Acceptance Criteria:**

**Given** a trained `StackedEnsemble`
**When** `ensemble.predict_proba(X)` is called with a pre-built feature DataFrame
**Then** for each stateless base model, `base_model.predict_proba(X[base_model.feature_names_])` is called using the stored post-fit feature name list
**And** for each stateful base model, `base_model.predict_proba(X)` is called (stateful models use `team_a_id`/`team_b_id` from metadata and ignore feature columns)
**And** base model predictions and `X[contextual_features]` are assembled into a meta-input DataFrame in the column order recorded during training
**And** `meta_learner.predict_proba(meta_X)` returns the final ensemble probability

**Given** a trained `StackedEnsemble` and a `data_dir` containing current-season data
**When** `ensemble.predict_bracket(data_dir, season)` is called
**Then** for each base model, a feature server is built from `base_model.feature_config` and current-season features are served
**And** base model predictions for all possible team matchups are generated
**And** the meta-input is assembled with contextual features from the current season
**And** a probability matrix (indexed by team_id pairs) is returned, suitable for passing to the Monte Carlo bracket simulator

**And** in both modes, the meta-learner input column order exactly matches the order recorded in the ensemble manifest, and a `ValueError` is raised if any required column is missing

**Source:** `specs/ensemble-architecture.md` §5
**Prerequisite:** Story 10.1

### Story 10.3: Dashboard and Model Registry Integration

As a **data scientist**,
I want to **see ensemble models in the dashboard leaderboard and inspect their components**,
So that **I can compare ensemble performance against single models and understand which base model the meta-learner is relying on**.

**Acceptance Criteria:**

**Given** a trained `StackedEnsemble` artifact in the output directory
**When** the user opens the Model Leaderboard dashboard page
**Then** the ensemble appears as a single entry with its `model_name`
**And** an expandable "Ensemble Components" section shows each base model's name and its OOF log loss (from the manifest)
**And** if the meta-learner supports `get_feature_importances()`, the importance chart shows `[pred_base_0, pred_base_1, ..., seed_diff, ...]` with interpretable labels (not raw column names)

**Given** the ensemble is selected in the dashboard
**When** the user navigates to the Bracket Visualizer page
**Then** `ensemble.predict_bracket(data_dir, season)` is called to generate the probability matrix, and the rest of the bracket visualizer works identically to single-model mode

**And** `StackedEnsemble` is registered in the model registry under the name provided to `run_training()` so the CLI `ncaa-eval evaluate` command works on ensemble run IDs

**Source:** `specs/ensemble-architecture.md` §5.1; `src/ncaa_eval/model/registry.py`; `dashboard/`
**Prerequisite:** Story 10.2

### Story 10.4: Ensemble Tutorial Notebook

As a **data scientist**,
I want to **follow a step-by-step tutorial that walks me through defining, training, and evaluating a custom ensemble**,
So that **I can understand the ensemble UX end-to-end and use it as a template for my own models**.

**Acceptance Criteria:**

**Given** the tutorial notebook `notebooks/tutorials/03_ensemble_model.ipynb`
**When** the user executes all cells in order
**Then** the notebook demonstrates:
  1. Importing base model classes from `ncaa_eval` and configuring each with feature-relevant kwargs
  2. Constructing a `StackedEnsemble` with at least one stateless model (XGBoost) and one stateful model (Elo) as base models, and a logistic regression meta-learner
  3. Calling `run_training(ensemble, ...)` with the one-liner UX
  4. Showing OOF log loss for each base model vs. the ensemble (demonstrates the blend adds value)
  5. Calling `ensemble.predict_bracket(data_dir, season)` to generate a bracket probability matrix
  6. Exporting a Kaggle submission CSV from the ensemble predictions
**And** all cells execute without error using the standard `ncaa_eval` conda env
**And** the notebook is referenced from `docs/tutorials.md` and included in the CI notebook-execution smoke test

**Source:** `specs/ensemble-architecture.md`; Story 7.9 (tutorial series); Story 9.1 (Kaggle export)
**Prerequisite:** Story 10.3

---

## Epic X: Cookiecutter Project Template

Extract NCAA_eval's project structure, toolchain, BMAD workflow configuration, and development conventions into a reusable cookiecutter template for future Python ML projects.

**Timing:** Post-project-completion — not scheduled for active development sprints. Epic X uses a letter designation (rather than a number) so that future numbered development epics (11, 12, 13...) always sort before it without requiring renumbering.

### Story X.1: Extract Cookiecutter Template Skeleton

As a **developer starting a new Python ML project**,
I want to **run `cookiecutter gh:dhilgart/NCAA_eval` and get a fully configured project scaffold**,
So that **I can skip the multi-day toolchain setup that NCAA_eval required and start with all quality gates, CI/CD, and BMAD workflows pre-wired**.

**Acceptance Criteria:**

**Given** the cookiecutter template at `template/{{cookiecutter.project_slug}}/`
**When** the developer runs `cookiecutter` with project name, slug, and author prompts
**Then** the generated project contains: Poetry `pyproject.toml` with standard quality gates (Ruff, mypy, pytest, nox), pre-commit config, GitHub Actions CI workflow, Sphinx docs scaffold, BMAD workflow configuration, and a minimal `src/` package structure
**And** `nox -s lint tests` passes on the freshly generated project
**And** the template is documented in `README.md` with a quickstart command

**Source:** `_bmad-output/planning-artifacts/epic-cookiecutter-template.md` (if it exists) or derive from Epic 1 and current project structure

---

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

### Game Theory Slider Implementation (Origin: Stories 7.5/7.7, 2026-02-28)

Interactive sliders for probability perturbation in the bracket visualizer and pool scorer. Two independent parameters: Upset Aggression (bidirectional temperature scaling) and Seed-Weight (linear blend with historical seed win rates). Spike research (Story 7.7) completed; implementation requires wiring slider transforms into `run_bracket_simulation()` and resolving UX spec collision (three sliders proposed vs. two recommended).

- **Effort:** Medium — core transformation functions (~150 lines), dashboard integration, slider-to-temperature mapping
- **Distinctness:** Novel UX feature; no current mechanism to perturb model probabilities before simulation
- **Source:** Story 7.7 spike — `specs/research/game-theory-slider-mechanism.md` §6, §8; UX Spec §3.1, §4.1
- **Deferred because:** Spike research completed but implementation requires PO approval on slider count (2 vs. 3) before scoping

### User-Editable Bracket (Origin: UX Spec Flow 1, 2026-02-28)

Allow users to manually override picks in the bracket visualizer, then re-score the custom bracket against MC simulations without modifying the underlying model. Requires bidirectional UI communication and session state management.

- **Effort:** Medium — bracket node click handlers, session state tracking, re-scoring pipeline
- **Distinctness:** Extends bracket from read-only visualization to interactive editing tool
- **Source:** UX Spec §3.1 (Flow 1: "Backtest-to-Selection" Diagnostic Loop, Step 4)
- **Deferred because:** Story 7.5 AC included "Team Detail Expansion" only; click-to-edit interaction was not scoped

### Metric Explorer: Round/Seed/Conference Drill-Downs (Origin: Story 7.4, 2026-02-28)

Extend the Model Deep Dive page's Metric Explorer to drill down by tournament round, seed matchup (1v16, 5v12, etc.), and conference. Currently only year-level drill-down is implemented.

- **Effort:** Medium — requires enriching `fold_predictions.parquet` with round/seed/conference columns, plus filter helpers and UI selectboxes
- **Distinctness:** Extends existing year-level drill-down to three additional dimensions
- **Source:** Story 7.4 AC #3 ("drill-down by year, round, seed matchup, or conference"); Dev Notes §Metric Explorer scope decision
- **Deferred because:** Data enrichment for round/seed/conference lookup was not in Story 7.4 AC scope; year drill-down delivered as MVP

### Candidate Entry Flagging (Origin: UX Spec Flow 1, 2026-02-28)

Allow users to flag specific bracket configurations (model + slider settings + bracket winners) as "candidate entries" for later comparison and export. Persists flags in session state.

- **Effort:** Low-Medium — candidate dataclass, session state persistence, flag button, summary page
- **Distinctness:** New workflow capability enabling iterative bracket refinement across multiple model runs
- **Source:** UX Spec §3.1 (Flow 1, Step 5: "User flags the specific configuration as a 'Candidate Entry'")
- **Deferred because:** Neither Story 7.5 nor 7.6 included flagging UI in their ACs

### CLI `predict` Command (Origin: PRD §3.3, 2026-02-28)

Command-line interface for generating per-game predictions, e.g., `ncaa predict --model elo --year 2025 --round "Round of 64" --output predictions.csv`. Complements the existing `train.py` CLI.

- **Effort:** Medium — CLI subcommand, prediction orchestrator, CSV output schema
- **Distinctness:** CLI pathway for predictions; currently only available via dashboard or notebook
- **Source:** PRD §3.3 (The CLI — "Background Jobs: Support for launching long-running backtests via CLI")
- **Deferred because:** Story 5.5 implemented training CLI only; prediction is currently served via dashboard and notebooks

### ~~Model Ensemble/Blending~~ → Promoted to Epic 10 (2026-03-09)

~~Probability ensemble combining multiple trained models' predictions via averaging, weighted voting, or stacking meta-learner. E.g., blend Elo + XGBoost predictions to reduce variance.~~

**Promoted:** Fully designed and broken into Epic 10 stories during PO decision session 2026-03-09. Architecture uses stacked generalization with a game-aware meta-learner (input-dependent weights). See `specs/ensemble-architecture.md` for the complete design.

### JSON Export for Pool Scorer (Origin: Story 7.6, 2026-02-28)

Extend Story 7.6 CSV export to also generate JSON format with nested structure including game predictions, round structure, and custom scoring configuration.

- **Effort:** Low — JSON schema mapping, export function, download button (~60 lines)
- **Distinctness:** Structured alternative to flat CSV for downstream integrations and programmatic access
- **Source:** Story 7.6 AC #5 (specifies CSV only)
- **Deferred because:** Story 7.6 AC required CSV export only; JSON is a natural enhancement for structured data consumers

### st.progress for Simulation (Origin: Story 7.6 / UX Spec §5.2, 2026-02-28)

Display real-time numeric progress bar (`st.progress()`) during Monte Carlo simulation instead of the current `st.spinner()`. Provides iteration count feedback during 1-5 second computation.

- **Effort:** Low — MC engine callback, `st.progress()` wrapper (~50 lines)
- **Distinctness:** UX polish replacing opaque spinner with quantitative progress
- **Source:** Story 7.6 AC #4 ("Simulation Progress"); UX Spec §5.2 (progress bar requirement)
- **Deferred because:** `st.spinner()` already prevents UI freezing; `st.progress()` requires MC engine to expose iteration callback

### Per-Game Prediction Explainability (Origin: PRD §3.2, 2026-02-28)

For each game in the bracket visualizer, display explainability metrics: feature importance contributions, confidence intervals, and natural language reasoning (e.g., "Elo gap favors team A by +8.5%").

- **Effort:** High — SHAP/LIME integration, confidence intervals, narrative generation, dashboard modal
- **Distinctness:** Per-game granularity vs. current model-level feature importance in Model Deep Dive
- **Source:** PRD §3.2 ("detailed views for specific models showing confusion matrices and feature importance")
- **Deferred because:** Story 7.4 implemented model-level feature importance only; per-game explainability requires SHAP (complex, slow) or custom heuristics

### Demo/Sample Data for Zero-Setup Onboarding (Origin: UX need, 2026-02-28)

Pre-package a small sample NCAA dataset (2-3 seasons, ~500 games) so new users can immediately run the full pipeline without waiting for `sync.py` to download 40+ years of data.

- **Effort:** Low — sample CSV generation, CLI `--use-sample-data` flag, Quick Start guide
- **Distinctness:** Developer experience feature enabling immediate pipeline execution after clone
- **Source:** PRD §3.3 (Usability: "3 commands" quick start); general onboarding best practice
- **Deferred because:** MVP pipeline requires actual data; sample data is a convenience feature for developer onboarding

### Custom Metric Plugin Registry (Origin: NFR3 / PRD §2, 2026-02-28)

Extend NFR3 extensibility to metrics: `@register_metric("matthews_correlation")` decorator allowing users to register custom evaluation metrics beyond the core set (Log Loss, Brier, AUC, ECE).

- **Effort:** Medium — metric registry singleton, Metric ABC, CLI/dashboard integration
- **Distinctness:** Completes NFR3 extensibility for the metrics axis (currently only model and scoring registries exist)
- **Source:** PRD §2 (NFR3: Extensibility — "models, scoring functions, metrics, feature generators"); Codebase audit P3-17
- **Deferred because:** NFR3 partially implemented (model + scoring registries); metric registry not scoped into any story

### Custom Feature Generator Plugin Registry (Origin: NFR3 / PRD §2, 2026-02-28)

`@register_feature_generator("my_feature")` decorator allowing users to register custom feature transformations beyond the core set (Elo, SRS, graph centrality). Requires temporal boundary validation to prevent data leakage.

- **Effort:** High — feature generator ABC, registry, pipeline integration, leakage prevention validation
- **Distinctness:** Completes NFR3 extensibility for the feature engineering axis; requires careful temporal boundary enforcement
- **Source:** PRD §2 (NFR3: Extensibility); Story 4.1 spike; Codebase audit P3-17
- **Deferred because:** NFR3 partially implemented; feature generator registry requires leakage prevention validation not scoped in any story

### Confusion Matrix in Model Deep Dive (Origin: PRD §3.2, 2026-02-28)

Display binary classification confusion matrix (TP/FP/TN/FN) with derived metrics (specificity, sensitivity, F1) in the Model Deep Dive page alongside existing reliability diagram.

- **Effort:** Low — `sklearn.metrics.confusion_matrix()`, heatmap rendering, dashboard section (~80 lines)
- **Distinctness:** Standard diagnostic complementing reliability diagram; useful for threshold analysis
- **Source:** PRD §3.2 ("confusion matrices and feature importance"); Story 7.4 (referenced but not in AC)
- **Deferred because:** Story 7.4 prioritized reliability diagram and feature importance; confusion matrix was not in final AC

### Public Bracket Competitive ROI Simulation (Origin: UX Spec Flow 2, 2026-02-28)

Simulate the user's bracket against public brackets generated from historical pick rates to estimate percentile rank in a public pool. Requires historical public bracket pick rate data not currently in the Kaggle dataset.

- **Effort:** High — historical pick rate data collection, public bracket generator, competitive ranking, dashboard visualization
- **Distinctness:** Fundamentally different from Pool Scorer: simulates against competitive brackets (other players' picks) rather than tournament outcomes
- **Source:** UX Spec §3.2 (Flow 2: "Pool-Specific ROI Simulation" — "simulates against 10,000 generated public brackets")
- **Deferred because:** Requires public bracket history data not available in Kaggle dataset; would need scraping or maintaining separate database

### run_training() API Refactor — PLR0913 Tech Debt (Origin: Story 8.1 review, 2026-03-04)

Bundle `run_training()` function's 7 keyword arguments into a `DateRange` dataclass (or similar) to resolve the `PLR0913` (too-many-arguments) suppression. Currently acknowledged via `# noqa: PLR0913` on the public API in `src/ncaa_eval/cli/train.py`.

- **Effort:** Low — dataclass definition (~15 lines) + signature refactor + call-site updates
- **Distinctness:** API design improvement; reduces cognitive load for callers of the public training API
- **Source:** Story 8.1 Senior Developer Review — "AC13 partial: run_training() retains # noqa: PLR0913 (7 keyword args — unchanged public API)"
- **Deferred because:** Public API change requires deliberate design decision; Story 8.1 prioritized module split over API refactor

### ESPN Marker-File Caching Metadata (Origin: Story 8.3, 2026-03-04)

Replace the current boolean marker file (`_espn_marker(year)`) with a `.espn_synced_{year}.json` metadata file recording `{ success_count, failed_count, timestamp }`. Currently `marker.touch()` runs even after partial ESPN fetch failures, permanently caching incomplete data.

- **Effort:** Low-Medium — metadata file schema, conditional marker write, pre-sync validation check
- **Distinctness:** Reliability improvement; prevents silently caching incomplete ESPN data
- **Source:** Story 8.3 Dev Notes — "marker-file caching design flaw; marker.touch() runs after partial failures"
- **Deferred because:** Story 8.3 addressed visibility (summary logging for partial failures) but not root cause; metadata file out of scope

### Dashboard `get_data_dir()` Path Fragility (Origin: Audit item 2.12, 2026-03-05)

Replace `Path(__file__).resolve().parent.parent.parent / "data"` in `dashboard/lib/filters.py` with a more robust path resolution (e.g., environment variable, configuration, or project root detection).

- **Effort:** Low — single function refactor (~10 lines)
- **Distinctness:** Maintenance improvement; prevents breakage if dashboard directory structure changes
- **Source:** Codebase audit item 2.12; `dashboard/lib/filters.py:56-58`
- **Deferred because:** Dashboard directory structure has been stable since Epic 7; low risk

### Undocumented Streamlit API Usage (Origin: Audit item 2.14, 2026-03-05)

Replace `event.selection.rows` (undocumented Streamlit dataframe selection API) with documented alternative or add pinned Streamlit version constraint to prevent breakage on upgrade.

- **Effort:** Low — API replacement or version pin (~20 lines)
- **Distinctness:** Maintenance improvement; prevents breakage on Streamlit upgrades
- **Source:** Codebase audit item 2.14; `dashboard/pages/1_Lab.py:116-129`
- **Deferred because:** API works correctly in current Streamlit version; will address if/when Streamlit breaks it

### Story 2.3 Open AI-Review Follow-ups (Origin: Audit item 2.17, 2026-03-05)

Address two deferred code quality items from Story 2.3's AI code review: (1) Add Pandera schema validation to KaggleConnector, (2) Replace `iterrows()` calls with vectorized operations in KaggleConnector.

- **Effort:** Medium — Pandera schema definition + vectorized CSV parsing (~200 lines)
- **Distinctness:** Code quality improvement; aligns ingest layer with project conventions
- **Source:** Codebase audit item 2.17; `src/ncaa_eval/ingest/connectors/kaggle.py`
- **Deferred because:** KaggleConnector works correctly; improvements are quality-of-code, not functional

### Test Helper Duplication: `_make_season_df` (Origin: Audit item 2.21, 2026-03-05)

Consolidate the duplicated `_make_season_df` helper function from `test_evaluation_splitter.py` and `test_evaluation_backtest.py` into a shared conftest fixture.

- **Effort:** Low — move to conftest, update imports (~15 lines)
- **Distinctness:** Minor test code quality improvement
- **Source:** Codebase audit item 2.21; `tests/unit/test_evaluation_splitter.py:18`, `tests/unit/test_evaluation_backtest.py:28`
- **Deferred because:** Duplication is minor; will consolidate when either test file is next modified

### Coverage Threshold Enforcement (Origin: Audit item P2-5, 2026-03-05)

Add `--cov-fail-under=XX` to CI pytest configuration to prevent silent coverage regression. Requires measuring current coverage level first to set an appropriate threshold.

- **Effort:** Low — measure coverage, add flag to CI config (~5 lines)
- **Distinctness:** Quality gate improvement; prevents coverage regression
- **Source:** Codebase audit item P2-5; `.github/workflows/python-check.yaml:31`
- **Deferred because:** Need to measure current coverage before setting threshold; arbitrary threshold risks blocking legitimate PRs

### Dashboard Quality Gate Inclusion (Origin: Audit item P2-6, 2026-03-05)

Add relaxed mypy configuration for `dashboard/` directory (e.g., `--follow-imports=normal` without `--strict`) to catch import errors and basic type mismatches while accommodating Streamlit's poor type stubs.

- **Effort:** Medium — mypy config section, fix existing type errors, CI integration
- **Distinctness:** Quality improvement for the primary user-facing layer
- **Source:** Codebase audit item P2-6; `noxfile.py:30-33`, `.pre-commit-config.yaml:67`
- **Deferred because:** Streamlit has poor type stubs; strict mypy impractical; relaxed config is a low-priority improvement

### NFR3 Tutorial — Clarify Metric Extension vs. Plugin Registry (Origin: Audit item P3-17, 2026-03-05)

Story 7.9 tutorial documents "How to Add a Custom Metric" via function injection into `run_backtest(metric_fns=...)`. This IS supported and the tutorial is functionally correct. However, audit item P3-17 flagged a potential reader confusion: NFR3 specifies a "plugin-registry architecture" for metrics, but the tutorial shows function-injection (not `@register_metric`). Clarify in the tutorial that metric extensibility uses function injection (not a registry) while scoring extensibility uses the `@register_scoring` decorator registry.

- **Effort:** Low — add a clarifying note or sidebar to the tutorial (~5 lines)
- **Distinctness:** Documentation clarity improvement; prevents confusion about which extension mechanism applies to metrics vs. scoring rules
- **Source:** Codebase audit item P3-17; `docs/tutorials/custom-metric.md`; PO decision: metric/feature-generator registries deferred, clarify tutorial distinction
