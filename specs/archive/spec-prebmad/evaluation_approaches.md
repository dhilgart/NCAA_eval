# NCAA Evaluation Approaches

This document defines the evaluation approaches that the NCAA evaluation UX needs to support, based on analysis of Kaggle March Machine Learning Mania competitions from 2014-2025.

## Primary Evaluation Metrics

### 1. Log Loss (2014-2022)

**Description**: Log Loss was the primary evaluation metric used in the March Machine Learning Mania competitions from 2014-2022.

**Formula**:
```
Log Loss = -1/n * Σ(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))
```

Where:
- `n` = number of games
- `y_i` = actual outcome (1 if team A wins, 0 if team B wins)
- `p_i` = predicted probability that team A wins

**Characteristics**:
- Heavily penalizes confident wrong predictions
- A single extreme wrong prediction (e.g., predicting 0.99 when outcome is 0) can result in a Log Loss of ~34.5 for that game
- Encourages conservative probability estimates
- Typical winning scores range from 0.5-0.7

**Implementation Requirements**:
- Handle edge cases for p_i = 0 or p_i = 1
- Support binary classification outcomes
- Provide per-game and aggregate scoring

### 2. Brier Score (2023-2025)

**Description**: Brier Score replaced Log Loss as the primary evaluation metric starting in 2023.

**Formula**:
```
Brier Score = 1/n * Σ(p_i - y_i)²
```

Where:
- `n` = number of games
- `y_i` = actual outcome (1 if team A wins, 0 if team B wins)
- `p_i` = predicted probability that team A wins

**Characteristics**:
- Less severe penalty for extreme wrong predictions compared to Log Loss
- Encourages more "gambling" with extreme probabilities
- Mean squared error of probability predictions
- Strictly proper scoring rule
- Range: 0 (perfect) to 1 (worst)

**Implementation Requirements**:
- Compute mean squared error of probabilities
- Support binary classification outcomes
- Provide per-game and aggregate scoring

## Secondary Evaluation Metrics

### 3. Accuracy

**Description**: Simple win/loss accuracy metric.

**Formula**:
```
Accuracy = (Number of correct predictions) / (Total number of predictions)
```

**Implementation Requirements**:
- Convert probabilities to binary predictions (typically threshold at 0.5)
- Support custom thresholds
- Provide overall and round-specific accuracy

### 4. Calibration Metrics

**Description**: Metrics to assess the calibration of probability predictions.

**Types**:
- **Reliability Diagram**: Visual assessment of calibration
- **Expected Calibration Error (ECE)**: Quantitative calibration measure
- **Brier Score Decomposition**: Separate uncertainty, reliability, and resolution components

**Implementation Requirements**:
- Bin predictions by probability ranges
- Compare predicted vs. actual frequencies
- Support different binning strategies

### 5. Ranking Metrics

**Description**: Metrics that evaluate the quality of team rankings rather than game predictions.

**Types**:
- **Kendall's Tau**: Rank correlation coefficient
- **Spearman's Rank Correlation**: Monotonic relationship assessment
- **Seed Performance**: How well predictions align with seed-based expectations

**Implementation Requirements**:
- Convert game predictions to team ratings/rankings
- Support different ranking aggregation methods
- Compare against seed-based baselines

## Tournament-Specific Evaluations

### 6. Round-by-Round Performance

**Description**: Evaluate model performance separately for each tournament round.

**Rounds**:
- Round 1 (64 teams → 32 teams)
- Round 2 (32 teams → 16 teams)
- Sweet 16 (16 teams → 8 teams)
- Elite 8 (8 teams → 4 teams)
- Final 4 (4 teams → 2 teams)
- Championship (2 teams → 1 team)

**Implementation Requirements**:
- Filter games by tournament round
- Compute metrics separately for each round
- Support both men's and women's tournaments

### 7. Upset Detection Performance

**Description**: Specifically evaluate model performance on predicting upsets.

**Definitions**:
- **Upset**: When a lower seed defeats a higher seed
- **Major Upset**: Significant seed differential (e.g., ≥ 4 seed difference)
- **Cinderella Stories**: Teams that advance further than seed expectations

**Implementation Requirements**:
- Identify upset games based on seed information
- Compute separate metrics for upset vs. non-upset games
- Support different upset definitions

### 8. Bracket Simulation Performance

**Description**: Evaluate how well models would perform in traditional bracket challenges.

**Approaches**:
- **Bracket Score**: Traditional bracket scoring systems
- **Expected Bracket Points**: Expected value across all possible brackets
- **Bracket Completion Rate**: How many games are correctly predicted in complete brackets

**Implementation Requirements**:
- Generate complete tournament brackets from game predictions
- Apply various bracket scoring systems
- Support different bracket completion strategies

## Advanced Evaluation Methods

### 9. Cross-Validation Strategies

**Description**: Time-based and tournament-specific cross-validation approaches.

**Types**:
- **Expanding Window CV**: Train on past years, test on future years
- **Leave-One-Tournament-Out**: Hold out entire tournaments for validation
- **Within-Tournament CV**: Use early rounds to predict later rounds

**Implementation Requirements**:
- Support temporal splitting of data
- Handle tournament structure in CV splits
- Provide performance trends over time

### 10. Ensemble Evaluation

**Description**: Evaluate performance of model ensembles and combinations.

**Methods**:
- **Simple Averaging**: Average probability predictions
- **Weighted Averaging**: Weight by historical performance
- **Stacking**: Meta-model to combine predictions
- **Bayesian Model Averaging**: Probabilistic model combination

**Implementation Requirements**:
- Support multiple ensemble methods
- Compare ensemble vs. individual model performance
- Track ensemble diversity and contribution

## Competition-Specific Features

### 11. Leaderboard Simulation

**Description**: Simulate competition leaderboards to understand relative performance.

**Features**:
- **Historical Leaderboards**: Compare against past competition results
- **Competition Modeling**: Model competitor behavior and submission strategies
- **Prize Probability**: Estimate likelihood of finishing in prize positions

**Implementation Requirements**:
- Access historical competition data
- Model competitor submission distributions
- Simulate multiple competition scenarios

### 12. Submission Strategy Evaluation

**Description**: Evaluate different submission strategies for competitions.

**Strategies**:
- **Conservative**: Safe, probability-focused submissions
- **Aggressive**: High-risk, high-reward probability adjustments
- **Game Theory**: Optimize against expected competitor behavior
- **Multiple Submissions**: Evaluate optimal submission portfolios

**Implementation Requirements**:
- Support different risk appetites
- Model competitor behavior
- Optimize submission portfolios

## Data Requirements

### Input Data Structure
- Game predictions with probabilities
- Tournament structure and bracket information
- Team seedings and rankings
- Historical game results
- Time-based data for temporal validation

### Output Requirements
- Metric scores with confidence intervals
- Per-game and aggregate performance
- Round-specific performance breakdowns
- Historical comparisons and benchmarks
- Visual performance summaries

## Implementation Considerations

### Performance Optimization
- Efficient computation for large tournament datasets
- Vectorized operations for metric calculations
- Caching of expensive computations
- Parallel processing for cross-validation

### User Experience
- Clear metric definitions and interpretations
- Interactive performance dashboards
- Customizable evaluation configurations
- Export capabilities for results

### Extensibility
- Plugin architecture for custom metrics
- Support for new evaluation methods
- Configurable metric weights and combinations
- API for integration with external tools

## References

- Kaggle March Machine Learning Mania Competitions (2014-2025)
- "Log Loss vs. Brier Score" - DRatings
- "March Machine Learning Mania 2023 Solution Writeup" - Medium
- "March Machine Learning Mania 1st Place Winner's Interview" - Kaggle Blog
- Scikit-learn documentation for evaluation metrics
