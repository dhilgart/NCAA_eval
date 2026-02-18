# Project Brief

**Project Name:** NCAA Tournament Data & Evaluation Platform
**Date:** February 13, 2026
**Status:** FINAL

## 1. Project Background & Opportunity
NCAA tournament modeling requires more than just static regression; it demands support for sequential systems (Elo), graph theory (PageRank), and opponent-adjusted metrics. However, current tools lack both the data infrastructure to support these complex models and the rigorous evaluation standards to judge them. This project will create a **unified platform** that serves as both the **canonical data warehouse** and the **rigorous evaluation engine** for all model types.

## 2. Goals & Objectives
* **Unified Data Platform:** Provide a single, standardized repository for raw and processed NCAA data, serving as the source of truth for both training and evaluation.
* **Support Advanced Modeling:** Native support for **Sequential** (Elo-style) and **Graph-based** (PageRank-style) models, alongside traditional "row-based" machine learning.
* **Hybrid Evaluation:** Assess models on both "Hard Science" (Calibration, Log Loss) and "Game Theory" (Bracket Scoring, Expected Points).
* **High Performance:** Architecture designed for speed (vectorization, parallelization, caching) to handle expensive cross-validation tasks.
* **Extensible Design:** A plugin-first architecture allowing users to inject custom metrics, scoring systems, and pre-processors.

## 3. Core Functional Requirements

### A. Data Platform (The "Feed")
* **Unified Data Ingestion:** Pipelines to fetch, clean, and store data from multiple sources (Kaggle, public APIs).
* **Advanced Feature Engineering Library:**
    * **Sequential transformers:** Generators for rolling averages, streaks, and "last N games" momentum features.
    * **Opponent Adjustment Engine:** Linear algebra solvers to calculate adjusted efficiency stats.
    * **Graph Builders:** Utilities to convert schedules into NetworkX graph objects.
* **Data Serving API:**
    * `get_chronological_season(year)`: Stream games in strict time order.
    * `get_training_data(years)`: Bulk fetch for vector models.
* **Caching:** Intelligent caching of expensive data fetches and pre-processing steps.

### B. Model Interface (The "Contract")
* **Abstract Base Class:** `Model` class with a flexible lifecycle:
    * **`train(data)`:** Receives standardized training data (sequential or batch).
    * **`predict(year, context)`:** Generates a full probability matrix ($N \times N$) for all matchups.
* **State Management:** Support for models that maintain internal state (e.g., Elo ratings) across the training sequence.

### C. Evaluation Engine (The "Referee")
* **Probabilistic Metrics:** Log Loss, Brier Score, ROC-AUC.
* **Calibration Analysis:**
    * **Metrics:** ECE (Expected Calibration Error).
    * **Visualizations:** Reliability Diagrams (Predicted vs. Actual probability scatter plots).
* **Tournament-Specific Scoring:**
    * **User-Defined Point Schedules:** Configurable scoring (e.g., default $2^{n-1}$, Fibonacci) for correct picks.
    * **Bracket Conversion:** Functionality to convert probability matrices into discrete brackets (e.g., "Max Likelihood") to calculate realized points.
    * **Expected Points:** Calculation of expected point totals based on raw probabilities.

### D. Validation & Workflow
* **Advanced Cross-Validation:**
    * **Sequential Splits:** Strict "walk-forward" validation for time-series models.
    * **Leave-One-Tournament-Out:** Native iterator for tournament-level backtesting.
* **Performance Optimization:**
    * **Vectorization:** Use `numpy` for all core metric calculations.
    * **Parallel Processing:** Multi-processing support for running Cross-Validation folds.
* **Leakage Prevention:** APIs designed to strictly prevent access to future games during training.

### E. Extensibility & UX
* **Plugin Architecture:** Registry system for custom metrics, scoring functions, and feature generators.
* **Visualization Suite:** Interactive dashboards for Reliability Diagrams, Year-by-Year accuracy plots, and Bracket Visualization.
* **Reporting:** Web-based (Streamlit) presentation layer for Model Leaderboards and deep-dive analysis.
* **Fail-Fast Debugging:** Deep logging, error traces, and data assertions. Custom verbosity levels.

## 4. Technical Constraints
* **Language:** Python 3.12+.
* **Core Libraries:** `numpy`, `pandas`, `scikit-learn`, `networkx`, `plotly`, `streamlit`.
* **Interface:** Code-based submission (Python classes only).

## 5. Success Metrics
* A user can implement a custom "Momentum-Elo" model using the platform's base classes and data feed, then run a parallelized 10-year backtest using "Fibonacci Scoring" in under 60 seconds.
* The system produces a "Calibration Report" that clearly distinguishes between "lucky" high-scoring models and "well-calibrated" probabilistic models.
