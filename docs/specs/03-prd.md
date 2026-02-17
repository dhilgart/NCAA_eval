# Product Requirements Document (PRD)

| **Project** | NCAA Tournament Data & Evaluation Platform |
| :--- | :--- |
| **Date** | February 13, 2026 |
| **Status** | **APPROVED** |
| **Author** | PM (John) |
| **Version** | 1.1 |

---

## 1. Goals and Background Context

### Goals
* **Unified Data Platform:** Establish a single, standardized repository for raw and processed NCAA data to serve as the definitive source of truth.
* **Advanced Modeling Support:** Enable native support for non-traditional models, specifically **Sequential** (Elo-style) and **Graph-based** (PageRank-style) systems, alongside standard "row-based" ML.
* **Hybrid Evaluation Engine:** Implement a dual-evaluation framework assessing "Hard Science" (Calibration, Log Loss) and "Game Theory" (Bracket Scoring, Expected Points).
* **High-Performance Architecture:** Design for speed using vectorization and parallelization to support expensive cross-validation tasks.
* **Extensible Design:** Deliver a plugin-first architecture allowing users to inject custom metrics, scoring systems, and feature generators.
* **Research-to-Dashboard Workflow:** Support a seamless transition from exploratory research (Jupyter) to high-level presentation (Streamlit).

### Background Context
Current tooling for NCAA tournament modeling is often fragmented. Data cleaning is repetitive, and evaluation is frequently limited to simple accuracy metrics that fail to capture the nuances of bracket pool betting (where calibration matters more than raw accuracy). Furthermore, most generic ML platforms cannot easily handle "Stateful" models (like Elo ratings) where predictions depend on the strict chronological sequence of previous games. This platform solves these infrastructure challenges so modelers can focus on strategy.

---

## 2. Requirements

### Functional Requirements

#### Data Engine & Ingestion
* **FR1 (Unified Data Ingestion):** The system must ingest, clean, and standardize raw NCAA data from multiple external sources into a unified internal schema.
* **FR2 (Persistent Local Store):** The system acts as a **Single-User Data Warehouse**. It must support a "One-Time Sync" command that fetches historical data and persists it locally (e.g., Parquet/SQLite). This local store acts as the authoritative Source of Truth for all downstream training and evaluation.
* **FR3 (Smart Caching):** The ingestion engine must implement a caching layer that strictly prefers valid local data over remote API calls to minimize latency and rate-limiting.
* **FR4 (Chronological Serving):** The Data API must support strict chronological streaming `get_chronological_season(year)` to support "walk-forward" training and prevent data leakage.

#### Feature Engineering & Modeling
* **FR5 (Advanced Transformations):** The platform must provide a library of transformations for:
    * **Sequential Features:** Rolling averages, streaks, and "last N games" momentum.
    * **Opponent Adjustments:** Linear algebra solvers for efficiency stats.
    * **Graph Representations:** Converting schedules into NetworkX graph objects for centrality metrics.
    * **Normalization:** Canonical mapping of diverse team names to single IDs.
* **FR6 (Flexible Model Contract):** The system must provide an abstract base class (`Model`) that supports:
    * **Stateless Models:** Standard batch training (e.g., XGBoost).
    * **Stateful Models:** Models that maintain internal state across a season (e.g., Elo ratings).

#### Evaluation & Validation
* **FR7 (Hybrid Evaluation Engine):** The evaluation system must calculate:
    * **Probabilistic Metrics:** Log Loss, Brier Score, ROC-AUC.
    * **Calibration Metrics:** ECE (Expected Calibration Error) and reliability diagrams.
    * **Tournament Scoring:** User-defined point schedules (e.g., Fibonacci) applied to simulated brackets.
* **FR8 (Validation Workflow):** The system must support "Leave-One-Tournament-Out" backtesting with strict temporal boundaries.
    * *Constraint:* The system must gracefully handle the 2020 "COVID Year" by allowing models to update state (training) without attempting to evaluate predictions (testing).
* **FR9 (Monte Carlo Tournament Simulator):** The system must implement a simulation engine capable of generating $N$ (configurable, default 10,000) realizations of the tournament bracket based on a model's probability matrix to calculate "Expected Points" and "Bracket Distribution" metrics.

### Non-Functional Requirements

* **NFR1 (Performance - Vectorization):** All core metric calculations must use vectorized operations (e.g., numpy) to minimize overhead during expensive cross-validation loops.
* **NFR2 (Performance - Parallelism):** The system must support parallel execution of cross-validation folds and model evaluations to maximize throughput on multi-core systems.
* **NFR3 (Extensibility):** The system must utilize a plugin-registry architecture to allow users to inject custom metrics, scoring functions, and feature generators without modifying core code.
* **NFR4 (Reliability - Leakage Prevention):** APIs must be architected to strictly enforce temporal boundaries, making it impossible for a model to access future game data during training.
* **NFR5 (Reliability - Fail-Fast Debugging):** The system must provide deep logging, error traces, and data assertions to facilitate debugging. Custom verbosity levels must be supported.

---

## 3. User Interface Design Goals

**Philosophy:** "Notebook-Native Research, Browser-Native Reports."

### 3.1 The "Lab" (Jupyter/Python API)
* **Interactive Visualization:** API methods (e.g., `model.plot_calibration()`) must return interactive **Plotly** figure objects that render directly in Jupyter notebooks.
* **Data Access:** Evaluation metrics and logs must be returned as rich **Pandas DataFrames** for ad-hoc slicing and dicing.
* **Inline Progress:** Display real-time progress bars within Jupyter cells for long-running training loops.

### 3.2 The "Presentation" (Streamlit Dashboard)
* **Leaderboard:** A sortable view comparing all trained models by various metrics.
* **Bracket Visualizer:** An interactive tournament tree view allowing users to inspect specific matchups and predicted probabilities.
* **Model Deep Dive:** Detailed views for specific models showing confusion matrices and feature importance.

### 3.3 The CLI (Command Line)
* **Background Jobs:** Support for launching long-running backtests via CLI (e.g., `python train.py --model elo --all-years`) that persist results to disk for later viewing in the Dashboard.
* **Progress Bars:** Display real-time progress bars showing, among other things, Cross-Validation fold progress.

### 3.4 The Documentation
* **API Docs:** Auto-generated from docstrings using `sphinx` and `furo` theme.
* **User Guide:** A comprehensive guide explaining the evaluation metrics, model types, and how to interpret the results.
* **Tutorials:** Step-by-step guides for common tasks (e.g., "How to create a custom model").

---

## 4. Technical Assumptions & Constraints

* **Language:** Python 3.12+ (Strict typing required).
* **Core Stack:**
    * **Data:** `pandas` (DataFrames), `numpy` (Vectorization), `scikit-learn` (Normalization, Scaling, Encoding, etc.).
    * **ML:** `xgboost` (Gradient Boosting), `scikit-learn` (Linear Models, etc.).
    * **Graph:** `networkx` (PageRank/Centrality).
    * **Parallelism:** `joblib` (Cross-validation loops).
    * **Visualization:** `jupyter` (Notebooks), `plotly` (Charts), `streamlit` (Dashboard).
* **Dev Tools:**
    * **Automated checking:** `pre-commit`
    * **Linting/Formatting:** `ruff`.
    * **Type Checking:** `mypy` (Strict mode).
    * **Testing:** `pytest`, `hypothesis`, `mutmut`.
    * **Documentation:** `sphinx` with `furo` theme.
    * **Manifest Checking:** `check-manifest`.
    * **Session Management:** `nox`.
    * **Dependency Management:** `edgetest`.
    * **Packaging:** `poetry`.
    * **Versioning:** `commitizen`.
* **Architecture:** Monolithic Python package (`ncaa_eval`) with a "Thin Client" Streamlit app (`dashboard/`).

---

## 5. Epic List & Details

### Epic 0: Project Initialization & Foundation
* **Story 0.1:** Configure repository including all dev tools.
* **Story 0.2:** Define code quality standards, style guide, pre-commit requirements, and PR checklists.
* **Story 0.3:** Define testing strategy (unit, integration, end-to-end, regression, mutation, etc.) including what tests to run at pre-commit vs. at PR time.
* **Story 0.4:** Set up versioning with Commitizen.
* **Story 0.5:** Set up packaging with Poetry.
* **Story 0.6:** Set up dependency management with Edgetest.
* **Story 0.7:** Set up session management with Nox.
* **Story 0.8:** Set up manifest checking with Check-manifest.
* **Story 0.9:** Set up automated testing with pytest.
* **Story 0.10:** Set up automated documentation generation with Sphinx.
* **Story 0.11:** Set up automated code formatting with Ruff.
* **Story 0.12:** Set up automated type checking with MyPy.
* **Story 0.13:** Set up automated code quality checking with pre-commit.

### Epic 1: Data Engine & Ingestion
* **Story 1.1 (Spike):** Evaluate data sources (Kaggle, KenPom, BartTorvik, ESPN, Nate Silver, etc.) for API feasibility and cost.
* **Story 1.2:** Public data connector: implement and verify API connections to each source.
* **Story 1.3:** Implement `python sync.py --source [kaggle|kenpom|barttorvik|espn|all] --dest <path>` script for fetching and caching public data to local Parquet/SQLite storage.

### Epic 2: Exploratory Data Analysis (EDA)
* **Story 2.1:** Perform initial data exploration of ingested data to understand data quality, structure, potential issues, relationships, histograms, etc.
* **Story 2.2:** Identify and document data quality issues and potential improvements.
* **Story 2.3:** Create data quality reports and recommendations for data cleaning.
* **Story 2.4:** Document data exploration findings and recommendations for feature engineering.

### Epic 3: Feature Engineering Suite
* **Story 3.1 (Spike):** Research feature engineering techniques for sports prediction (e.g. opponent adjustments, advanced stats, home/away effects, strength of schedule, recent form, etc.).
* **Story 3.2:** Implement Canonical Team ID mapping and name normalization.
* **Story 3.3:** Implement data cleaning, scaling, categorical encoding, and normalization.
* **Story 3.4:** Build "Graph Builders" to convert season schedules into NetworkX objects.
* **Story 3.5:** Implement "Sequential Transformations" for rolling averages and momentum features.
* **Story 3.6:** Implement stateful feature serving to feed data to stateful models in training
* **Story 3.7:** Implement techniques identified in Story 3.1.

### Epic 4: Core Modeling Framework
* **Story 4.1 (Spike):** Research types of models that have been used for this task by other people. Especially from kaggle march machine learning mania discussion boards from the various years. Document requirements to enable the `Model` Abstract Base Class (ABC) to support all of those modeling approaches.  
* **Story 4.2:** Create the `Model` Abstract Base Class (ABC) enforcing `train`, `predict`, and `save`.
* **Story 4.3:** Implement Reference **Stateful Model** (Elo) handling season-to-season state persistence.
* **Story 4.4:** Implement Reference **Stateless Model** (XGBoost wrapper).

### Epic 5: Evaluation & Validation Engine
* **Story 5.1:** Implement "Walk-Forward" Cross-Validation Splitter.
    * *Critical:* Must handle 2020 by yielding training data but skipping evaluation.
* **Story 5.2:** Build the Vectorized Metric Library (Log Loss, Brier, ECE) using `numpy` and `scikit-learn`.
* **Story 5.3 (Spike):** Research how we can improve our confidence in our tournament simulation predictions given the limited data.
* **Story 5.4:** Implement the Tournament Simulator (Bracket generation via Max Likelihood or Monte Carlo) in light of the outcomes of Story 5.3.

### Epic 6: Lab & Presentation Layers
* **Story 6.1:** Build Plotly adapters for inline Jupyter visualization.
* **Story 6.2:** Develop Streamlit Dashboard for Model Leaderboards.
* **Story 6.3:** Develop Streamlit Bracket Visualizer.

---

## 6. Success Metrics
1.  **Performance:** A 10-year parallelized backtest of a standard Elo model (training & inference only) must complete in under **60 seconds**. (Excludes Monte Carlo simulation time).
2.  **Calibration:** The platform must auto-generate a Reliability Diagram that clearly identifies if a model is over-confident.
3.  **Usability:** A new developer can clone the repo and run a full pipeline (sync -> train -> eval) via CLI in under 3 commands.
