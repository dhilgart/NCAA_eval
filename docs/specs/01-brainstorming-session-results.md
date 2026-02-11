# Brainstorming Session Results

**Session Date:** Wednesday, February 11, 2026
**Facilitator:** Business Analyst Mary
**Participant:** User

## Executive Summary

**Topic:** Architecture for a Python repository to evaluate NCAA tournament prediction models.

**Session Goals:** Define functional capabilities, standard interfaces, and evaluation metrics for a static model evaluation system.

**Techniques Used:** What If Scenarios, First Principles Thinking, Morphological Analysis, Role Playing, Resource Constraints.

**Total Ideas Generated:** ~15 key functional capabilities.

**Key Themes Identified:**
- **Probabilistic Focus:** The atomic unit of prediction is a probability matrix, not a bracket.
- **Data Integrity:** Strict normalization and handling of diverse data sources is required.
- **Rigorous Validation:** "Leave-One-Tournament-Out" cross-validation is the critical priority over simulation.
- **Debugging First:** The system must be "loud" about errors to aid model developers.

---

## Technique Sessions

### Technique 1: What If Scenarios - 10 min
**Description:** Explored edge cases regarding specific evaluation filters, prediction types, and data conflicts.

**Ideas Generated:**
1. **Scenario A (Upsets):** Capability to filter evaluation datasets by specific criteria (e.g., seed difference > 5, specific rounds) to isolate model performance on upsets.
2. **Scenario B (Comparison):** Constraint that all models must output probabilistic predictions; binary "win/loss" predictions are not supported to ensure fair comparison.
3. **Scenario C (Data Conflicts):** Requirement for a "Standardized Team Name Normalization & Resolution" layer to map disparate sources (Kaggle vs. KenPom) to a single schema.

**Insights Discovered:**
- The system cannot be agnostic to data quality; it must actively manage ingestion.
- Comparative evaluation requires a standardized probabilistic output format.

### Technique 2: First Principles Thinking - 5 min
**Description:** Stripped the problem down to its fundamental atomic units.

**Ideas Generated:**
1. **Atomic Unit:** The fundamental output is a **Full Matchup Probability Matrix** (Team A vs. Team B for all $N*(N-1)/2$ combinations), not a "bracket" path.
2. **Evaluation Basis:** Models are evaluated against reality (Ground Truth) using this matrix, regardless of "bracket busting" in early rounds.

**Insights Discovered:**
- This simplifies the interface definition: the return type of any model class must be a probability matrix.

### Technique 3: Morphological Analysis - 10 min
**Description:** Broke the system into dimensions (Data, Metrics, Interface) to find combinations.

**Ideas Generated:**
1. **Data Sources:** Kaggle Data, KenPom/Torvik (Advanced Stats), requiring research into additional sources.
2. **Metrics:**
   - **Accuracy/Calibration:** Log Loss, Brier Score.
   - **Ranking:** ROC-AUC.
   - **Context-Aware:** Performance in "High Upset" vs. "Low Upset" years.
   - **Validation:** Time-based split, Leave-One-Tournament-Out.
3. **Interface:** Strict Python Class `predict()` method (No CSV uploads allowed).

### Technique 4: Role Playing - 10 min
**Description:** Adopted "The Forensics Expert" and "The Portfolio Manager" personas to uncover missing features.

**Ideas Generated:**
1. **Forensics Expert (Debug):** System must provide detailed error traces, data logging (inputs/outputs), and fail-fast assertions to help users diagnose why a specific historic tournament crashed their model.
2. **Portfolio Manager (Risk):** The system will *not* manage portfolios/ensembles natively. Ensembling is a user-side concern to keep the interface clean.

### Technique 5: Resource Constraints - 5 min
**Description:** Forced prioritization by assuming a strict 2-week deadline for v1.0.

**Ideas Generated:**
1. **The "Hill to Die On":** The **Advanced Cross-Validation Engine** (Leave-One-Tournament-Out) is Priority #1.
2. **Deferred:** The "Tournament Simulator" (Monte Carlo prize estimation) is pushed to v1.1 or later.

---

## Idea Categorization

### Immediate Opportunities
*Ideas ready to implement now*

1.  **Standardized Model Interface**
    - Description: Abstract Base Class requiring a `predict()` method returning a probability matrix.
    - Why immediate: Core dependency for all other features.
    - Resources needed: Python abstract base class definitions.

2.  **Advanced Cross-Validation Engine**
    - Description: Native support for time-series splits and Leave-One-Tournament-Out.
    - Why immediate: Critical for preventing overfitting; the scientific foundation of the repo.
    - Resources needed: Custom iterator logic for tournament years.

3.  **Data Normalization Layer**
    - Description: Mapping system to resolve team name discrepancies across sources.
    - Why immediate: Essential for multi-source data ingestion.
    - Resources needed: Mapping dictionary/database of NCAA team names.

### Future Innovations
*Ideas requiring development/research*

1.  **Tournament Simulator**
    - Description: Monte Carlo engine to simulate 10,000+ pools for prize likelihood.
    - Development needed: Complex simulation logic and scoring rules engine.
    - Timeline estimate: v1.1 or v2.0.

---

## Action Planning

### Top 3 Priority Ideas

#### #1 Priority: Advanced Cross-Validation Engine
- **Rationale:** Without robust backtesting (specifically avoiding look-ahead bias), the evaluation is scientifically worthless.
- **Next steps:** Define the iterator interface for `LeaveOneTournamentOut` and `TimeSeriesSplit`.
- **Resources needed:** Historical tournament result data.
- **Timeline:** Sprint 1.

#### #2 Priority: Standardized Probability Matrix Interface
- **Rationale:** The "First Principles" core of the system. Defines the contract between user code and evaluation logic.
- **Next steps:** Draft the `Model` abstract base class and `PredictionResult` data class.
- **Resources needed:** None (pure code design).
- **Timeline:** Sprint 1.

#### #3 Priority: Data Ingestion & Normalization
- **Rationale:** "Garbage in, garbage out." We need clean data to feed the models.
- **Next steps:** Research data sources (Kaggle/KenPom) and build the name resolution map.
- **Resources needed:** Access to raw data APIs or exports.
- **Timeline:** Sprint 1-2.

---

## Reflection & Follow-up

**What Worked Well:**
- The **Resource Constraint** technique clarified that Cross-Validation is more critical than Simulation for the MVP.
- **First Principles** thinking effectively shifted the focus from "brackets" to "probability matrices."

**Areas for Further Exploration:**
- Researching specific additional data sources beyond Kaggle/KenPom.
- Defining the exact API for the "Context-Aware" evaluation (e.g., how do we mathematically define a "High Upset Year"?).

---
*Session facilitated using the BMAD-METHOD™ brainstorming framework*