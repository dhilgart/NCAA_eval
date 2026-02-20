# Story 4.1: Research Feature Engineering Techniques (Spike)

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want a documented survey of feature engineering techniques used in sports prediction (especially NCAA tournament contexts),
so that I can make informed decisions about which transformations to implement based on proven approaches and EDA findings.

## Acceptance Criteria

1. **Given** EDA findings (Epic 3) have identified promising signals and the project needs feature engineering direction, **When** the data scientist reviews the spike findings document, **Then** opponent adjustment methods are documented (e.g., ridge regression efficiency, SRS-style solvers).
2. **And** sequential/momentum feature approaches are catalogued (rolling averages, streaks, recency weighting).
3. **And** graph-based features are surveyed (PageRank, betweenness centrality, clustering coefficient).
4. **And** Kaggle March Machine Learning Mania discussion boards are reviewed for community-proven techniques.
5. **And** each technique is assessed for feasibility, complexity, and expected predictive value.
6. **And** a prioritized implementation plan is documented, aligned with Stories 4.2–4.7 scope.
7. **And** the findings are committed as `specs/research/feature-engineering-techniques.md`.
8. **And** the product owner reviews the spike findings and approves the prioritized implementation plan before downstream Stories 4.2–4.7 begin. *(Decision-gate: the spike produces recommendations; the product owner makes the final MVP-scope selections.)*

## Tasks / Subtasks

- [ ] Task 1: Pre-Spike Context Loading (AC: 1–7) — read ALL required documents before starting research
  - [ ] 1.1: Read `notebooks/eda/eda_findings_synthesis.md` — primary EDA reference; contains 13 ranked feature candidates and Epic 4 story-by-story guidance
  - [ ] 1.2: Read `notebooks/eda/statistical_exploration_findings.md` Section 7 — empirically derived normalization transforms for all 18 box-score stats (do NOT re-derive; these are already solved)
  - [ ] 1.3: Read `specs/research/data-source-evaluation.md` — established research document format/convention to follow
  - [ ] 1.4: Review `specs/05-architecture-fullstack.md` Section 3 (Tech Stack) and Section 9 (Project Structure) — confirms `networkx` is already a dependency; `transform/` is the target module

- [ ] Task 2: Research Opponent Adjustment Techniques (AC: 1, 5, 6)
  - [ ] 2.1: Research SRS (Simple Rating System) — least-squares approach; compare to KenPom's AdjEM
  - [ ] 2.2: Research ridge regression / Lasso for efficiency rating (AdjOE / AdjDE) — penalized least squares on game outcomes
  - [ ] 2.3: Research Massey Rating system (linear algebra approach) — already available via `MMasseyOrdinals.csv` with 100+ systems
  - [ ] 2.4: Assess feasibility vs. complexity vs. expected predictive value; compare against EDA SoS baseline (r=0.2970)
  - [ ] 2.5: Validate: opponent-adjusted efficiency should exceed raw SoS (r=0.2970) baseline — if not, the adjustment adds no value

- [ ] Task 3: Research Sequential / Momentum Feature Approaches (AC: 2, 5, 6)
  - [ ] 3.1: Research rolling window sizes — candidates: last 5, 10, 20 games (EDA has no strong evidence; must be empirically validated in Story 4.4)
  - [ ] 3.2: Research recency-weighting schemes — exponential decay vs. uniform window vs. adaptive window
  - [ ] 3.3: Research streak features — win/loss streak length, direction, and termination points
  - [ ] 3.4: Research performance trajectory features (momentum) — rate-of-change of rolling averages
  - [ ] 3.5: Review Kaggle MMLM top solutions for sequential feature approaches

- [ ] Task 4: Research Graph-Based Features (AC: 3, 5, 6)
  - [ ] 4.1: Research PageRank on win/loss directed graph (W→L edge, weight = margin of victory) — compare to naive SoS (r=0.2970 baseline)
  - [ ] 4.2: Research betweenness centrality — quantifies "bridge" teams in schedule network
  - [ ] 4.3: Research HITS (hub/authority) algorithm for schedule-strength quantification
  - [ ] 4.4: Research incremental graph construction — must support walk-forward updates (cannot recompute from scratch at each time step)
  - [ ] 4.5: Assess whether graph features justify extra complexity (networkx is already in tech stack — no new dependency)
  - [ ] 4.6: Evaluate graph feature predictive value against SoS baseline — REQUIRED validation before recommending inclusion

- [ ] Task 5: Review Kaggle MMLM Community Techniques (AC: 4, 5, 6)
  - [ ] 5.1: Review Kaggle MMLM 2024, 2023, 2022, 2021 competition discussion boards (public threads)
  - [ ] 5.2: Extract community-proven feature engineering patterns — specifically noting which techniques appear consistently across winning solutions
  - [ ] 5.3: Review published MMLM writeups/kernels for feature importance analysis
  - [ ] 5.4: Identify techniques in MMLM that are NOT yet captured in the EDA tier list — add as new candidates

- [ ] Task 6: Research Massey Ordinal Systems (AC: 4–6)
  - [ ] 6.1: Identify top Massey Ordinal systems by season coverage from `MMasseyOrdinals.csv` — known top 5: AP, DOL, COL, MOR, POM (all 23 seasons)
  - [ ] 6.2: Research which Massey systems are most widely cited in MMLM top solutions
  - [ ] 6.3: Research composite ranking approaches (mean, weighted ensemble of multiple Massey systems)
  - [ ] 6.4: Assess: do Massey Ordinals add signal beyond box-score features already in the EDA tier list?

- [ ] Task 7: Compile Findings & Create Research Document (AC: 1–8)
  - [ ] 7.1: Create `specs/research/feature-engineering-techniques.md` following Story 2.1 document conventions
  - [ ] 7.2: Section 1 — Data Context (what EDA already confirmed; what remains open; cite specific r-values and finding sources)
  - [ ] 7.3: Section 2 — Opponent Adjustment Techniques survey with feasibility/complexity/value assessment
  - [ ] 7.4: Section 3 — Sequential/Momentum Feature survey with window-size recommendations and implementation approach
  - [ ] 7.5: Section 4 — Graph Feature survey with centrality comparison vs. SoS baseline
  - [ ] 7.6: Section 5 — Massey Ordinal analysis and recommendation
  - [ ] 7.7: Section 6 — Community Techniques from Kaggle MMLM review
  - [ ] 7.8: Section 7 — Prioritized Implementation Plan mapping each technique to Story 4.2–4.7 and flagging which are MVP vs. post-MVP
  - [ ] 7.9: DO NOT make final MVP-scope selections — present ranked options with trade-offs for product owner review (see AC 8)

- [ ] Task 8: Commit & Gate (AC: 7, 8)
  - [ ] 8.1: `git add specs/research/feature-engineering-techniques.md`
  - [ ] 8.2: Commit: `docs(research): feature engineering techniques survey (Story 4.1)`
  - [ ] 8.3: Update sprint-status.yaml: `4-1-research-feature-engineering-techniques` → `done`

## Dev Notes

### Story Nature: Pure Documentation Research Spike — No Code

This is a **research/documentation spike** like Story 2.1. The primary deliverable is a Markdown findings document. **No code goes into `src/ncaa_eval/` during this story.** No new tests. No Parquet changes.

`mypy --strict`, Ruff, and the vectorization mandate do NOT apply to research documents. No `from __future__ import annotations` required.

### Critical Pre-Spike Context — READ FIRST

**`eda_findings_synthesis.md` is the MANDATORY starting point.** EDA already resolved many open questions:

| Open Question | Status |
|---|---|
| Which box-score stats correlate with tournament outcomes? | ✅ RESOLVED — 13 features ranked in 3 tiers with Pearson r values |
| What normalization transform per stat? | ✅ RESOLVED — full table in `statistical_exploration_findings.md` Section 7 |
| Which Massey Ordinal systems have best coverage? | ✅ RESOLVED — AP, DOL, COL, MOR, POM (23 seasons each) |
| Do graph features add value beyond SoS? | ❌ OPEN — priority research question #1 |
| Optimal rolling window size? | ❌ OPEN — EDA found no strong evidence; needs empirical validation |
| Which Massey systems add signal beyond box scores? | ❌ OPEN — priority research question #2 |

**Do NOT re-derive EDA statistics.** All correlation values and normalization recommendations are settled — cite them by reference.

### CRITICAL: Decision-Gate Requirement (from template-requirements.md)

**Template pattern from Story 2.1 Code Review Round 3:**
> Spike stories produce recommendations. Decisions belong to the product owner. The dev agent must present options with trade-offs, **not commit selections unilaterally**.

The findings document MUST:
- Present each technique with its trade-offs (complexity, dependency, expected value)
- Explicitly mark techniques as "Recommended for MVP", "Recommended for Post-MVP", or "Not Recommended" — but frame these as proposals, not decisions
- NOT update epics.md or constrain downstream stories until product owner approval (AC 8)

### Output File Location

**Create:** `specs/research/feature-engineering-techniques.md`

Place alongside `specs/research/data-source-evaluation.md` (Story 2.1 output) for consistency.

### EDA-Confirmed Baselines to Beat

Any new technique must justify its complexity by exceeding these empirical baselines:

| Baseline | Metric | Evidence |
|---|---|---|
| Raw SoS (opponent win rate) | Pearson r = 0.2970 with tournament advancement | `statistical_exploration_findings.md` Section 4 |
| FGM | r = 0.2628 | Section 5 |
| FGPct | r = 0.2269; tournament winner diff = +0.078 | Section 5 |
| Scoring (WScore) | r = 0.2349 | Section 5 |
| PF (negative) | r = -0.1574 | Section 5 |

Graph centrality features (PageRank, betweenness) must be validated to exceed r=0.2970 before being recommended for Story 4.5. Opponent-adjusted efficiency must exceed r=0.2970 before being recommended for Story 4.6.

### Data Source Context for Research

Available in the local Parquet store (populated via Stories 2.2–2.4):

| File | Rows | Coverage | Relevance |
|---|---|---|---|
| `MRegularSeasonDetailedResults.csv` | 118,882 | 2003–2025 | Box scores (FGM, FGA, 3P, FT, OR, DR, Ast, TO, Stl, Blk, PF) |
| `MNCAATourneyDetailedResults.csv` | 1,382 | 2003–2024 | Tournament box scores (2025 not available until tourney completes) |
| `MMasseyOrdinals.csv` | ~500K+ | varies per system | 100+ ranking systems; top 5 by coverage: AP, DOL, COL, MOR, POM |
| `MNCAATourneySeeds.csv` | 2,626 | 1985–2025 | Seed number, region, play-in flag |
| `MTeamConferences.csv` | ~9,000+ | per season | Conference membership by (season, team_id) |
| `MTeamSpellings.csv` | 1,177 | all years | Canonical name mapping for 380 teams |

**Box-score coverage gap:** Pre-2003 data has only compact results (W/L, score, loc). Any feature requiring FGM/FGA will be limited to 2003+. The research should note whether a technique can gracefully degrade to compact-only data for 1985–2002.

**2025 deduplication required:** 4,545 games stored twice (Kaggle + ESPN). Any calculation touching 2025 must deduplicate first. See `eda_findings_synthesis.md` Section 1 for the deduplication code pattern.

### Known Techniques to Survey (Minimum Coverage)

From epics.md Epic 4 requirements and EDA synthesis — these MUST appear in the research document:

**Opponent Adjustments (Story 4.6 scope):**
- Simple Rating System (SRS) — iterative least-squares; compare to KenPom AdjEM
- Ridge regression / penalized least squares on efficiency stats (AdjOE, AdjDE)
- Massey's method — same linear algebra, different matrix formulation
- Assess: which solver best handles teams with few games (new teams, unusual schedules)?

**Sequential Features (Story 4.4 scope):**
- Rolling window averages (uniform weights) — window sizes to evaluate: 5, 10, 20 games
- Exponentially weighted moving averages (EWMA) — alpha parameter selection
- Streak features — current win/loss streak length
- Momentum — rate-of-change (derivative) of rolling stats over time

**Graph Features (Story 4.5 scope):**
- PageRank (directed, weighted by margin of victory) — `networkx.pagerank()`
- Betweenness centrality — `networkx.betweenness_centrality()`
- Clustering coefficient — `networkx.clustering()`
- HITS (hub/authority scores) — `networkx.hits()`
- Incremental graph update strategy for walk-forward compatibility

**Massey Ordinal Systems (Story 4.3 scope):**
- Which systems to include as features vs. validation benchmarks?
- Composite ranking: mean, weighted average, PCA-reduced
- Systems with highest MMLM community usage

**Community Techniques (Kaggle MMLM):**
- Elo rating variants — variable K-factor, home-court adjustment, margin-of-victory scaling
- Glicko-2 / TrueSkill ratings — uncertainty quantification
- Time-decay features — weight recent games more in pre-tournament features
- Ensemble ranking composites

### Architecture Guardrails

**Package location:** Feature engineering code goes in `src/ncaa_eval/transform/` (currently empty `__init__.py`).

**Architecture constraints (from `specs/05-architecture-fullstack.md`):**
- All feature transformations must be **vectorized** (numpy/pandas — no for-loops over DataFrames; Architecture Section 12, NFR1)
- `mypy --strict` mandatory — all code in `src/ncaa_eval/transform/` must be type-annotated
- `from __future__ import annotations` required in all Python files (Ruff enforced)
- Input DataFrames in, feature DataFrames out — functional pipeline pattern (Architecture Section 8.1)
- NetworkX is already in the tech stack (`specs/05-architecture-fullstack.md` Section 3 Tech Stack) — no new dependency needed for graph features

**Normalization configurability (from `eda_findings_synthesis.md` Section 4, Story 4.7):**
Feature serving layer must support:
- `gender_scope: Literal["separate", "combined"] = "separate"` — normalization statistics differ between M/W data
- `dataset_scope: Literal["regular_season", "tournament", "combined"] = "regular_season"` — survivorship bias in tournament data means separate normalization is required

The research document should address which techniques require separate normalization by gender/dataset scope.

### Normalization Is Already Resolved — Do NOT Reopen

`statistical_exploration_findings.md` Section 7 empirically determined normalization transforms for all 18 box-score stats across 162 observations. The research document should reference this table, not derive new recommendations:

| Stat Group | Transform | Scaler |
|---|---|---|
| Bounded rates (FGPct, 3PPct, FTPct, TO_rate) | logit | StandardScaler |
| Right-skewed counts (Blk, Stl, TO, FTM, OR, FGM3, FTA) | sqrt | RobustScaler |
| High-volume counting (Score, FGM, FGA, DR, Ast, FGA3, PF) | none | StandardScaler |

Story 4.1 research should only address normalization for NEW features not in this table (e.g., graph centrality metrics, Elo ratings, Massey ordinals).

### Previous Story Learnings (Story 2.1 Spike — Data Source Evaluation)

Pattern to follow from Story 2.1 (`data-source-evaluation.md`):
- Each technique: name, description, access method, expected value, complexity, risk/caveats
- Decision sections clearly separate "Recommended for MVP" from "Deferred" — but frame as proposals, not decisions
- Use a summary table at the top for quick navigation
- Do NOT live-test or implement anything during the research spike — document theoretical approach and feasibility

**Lessons from Story 2.1 code review (CRITICAL):**
- ❌ Do not select techniques for MVP unilaterally — present ranked options with trade-offs for product owner approval
- ❌ Do not include techniques not validated during the spike — if you can't assess a technique's value, mark it "Requires Validation" rather than recommending it
- ✅ Rate-limit awareness equivalent: document computational cost (O(n) vs. O(n²)) for each feature type — some graph algorithms are expensive at scale

### What NOT To Do

- **Do not** create Python code or modify any `src/ncaa_eval/` files
- **Do not** re-run EDA notebooks or recompute statistics — cite existing findings by file and section
- **Do not** finalize MVP scope unilaterally — present options with trade-offs and let the product owner decide (AC 8)
- **Do not** reopen the normalization question — Section 7 of `statistical_exploration_findings.md` already resolves it
- **Do not** place the findings document in `_bmad-output/` — use `specs/research/` (same directory as Story 2.1 output)
- **Do not** add the research document to mypy scope or Ruff checks

### Project Structure Notes

- New file: `specs/research/feature-engineering-techniques.md`
- No changes to `src/`, `tests/`, `pyproject.toml`, or any existing files
- Commit type: `docs(research): ...` (same pattern as Story 2.1)
- `specs/` is tracked by git (not gitignored) — commit directly
- Branch: currently on `main` (Epic 4 starts fresh — no existing story branch to merge)

### References

- [Source: notebooks/eda/eda_findings_synthesis.md — All 4 sections; primary context for this spike]
- [Source: notebooks/eda/statistical_exploration_findings.md — Section 7 (box-score normalization, already resolved)]
- [Source: notebooks/eda/statistical_exploration_findings.md — Section 4 (SoS baseline r=0.2970)]
- [Source: notebooks/eda/statistical_exploration_findings.md — Section 5 (top correlation values)]
- [Source: _bmad-output/planning-artifacts/epics.md#Epic 4 — All Story 4.1–4.7 scope definitions]
- [Source: specs/05-architecture-fullstack.md#Section 3 — Tech Stack (networkx in stack; vectorization mandate)]
- [Source: specs/05-architecture-fullstack.md#Section 9 — Unified Project Structure (transform/ module)]
- [Source: specs/research/data-source-evaluation.md — Document format convention from Story 2.1]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — Story 2.1 Code Review Round 3 lessons on decision-gate requirement]

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6 (create-story workflow)

### Debug Log References

### Completion Notes List

### File List
