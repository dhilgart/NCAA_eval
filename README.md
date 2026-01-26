[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg?style=flat-square)](https://conventionalcommits.org)
[![Github Actions](https://github.com/dhilgart/NCAA_eval/actions/workflows/python-check.yaml/badge.svg)](https://github.com/dhilgart/NCAA_eval/actions/workflows/python-check.yaml)

# NCAA Eval

A comprehensive Python repository for evaluating NCAA tournament prediction models. This project provides standard interfaces for prediction models and evaluation methods, enabling systematic assessment of March Madness bracket predictions.

## Overview

NCAA Eval is designed to support the evaluation of various prediction models used in NCAA March Madness tournaments, particularly those competing in Kaggle's March Machine Learning Mania competitions. The repository offers:

- **Standard Model Interface**: A well-defined interface that prediction models must implement
- **Standard Evaluation Interface**: A flexible framework for different evaluation methodologies
- **Extensible Architecture**: Support for both built-in and custom evaluation methods
- **Simple UX**: User-friendly interfaces for model integration and evaluation selection

## Features

### Model Interface
- Standardized interface supporting all typical model types found in Kaggle competitions
- Compatible with various prediction approaches (probabilistic, categorical, regression-based)
- Easy integration for new models through defined contracts

### Evaluation Methods
- Built-in evaluation methods scraped from Kaggle March Machine Learning Mania competitions
- Support for multiple evaluation metrics (log loss, Brier score, accuracy, etc.)
- Flexible selection of evaluation methods
- Extensible framework for custom evaluation implementations

### Development Tools
- **Code Quality**: Ruff for linting and formatting
- **Testing**: Pytest for comprehensive test coverage
- **Data Validation**: Pydantic for robust data validation
- **Import Management**: isort for consistent import sorting
- **Pre-commit Hooks**: Automated code quality checks
- **Dependency Management**: edgetest for managing dependencies
- **Type Checking**: mypy for static type analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/dhilgart/NCAA_eval.git
cd NCAA_eval

# Install dependencies
pip install -e .

# Install pre-commit hooks
pre-commit install
```

## Quick Start

### Basic Usage

```python
from ncaa_eval import ModelEvaluator, YourPredictionModel

# Initialize your model (must implement the standard interface)
model = YourPredictionModel()

# Create evaluator with default evaluation methods
evaluator = ModelEvaluator()

# Run evaluation
results = evaluator.evaluate(model)

# View results
print(results.summary())
```

### Custom Evaluation Methods

```python
from ncaa_eval import ModelEvaluator, CustomEvaluationMethod

# Define custom evaluation method
custom_eval = CustomEvaluationMethod()

# Use specific evaluation methods
evaluator = ModelEvaluator(evaluation_methods=[custom_eval])
results = evaluator.evaluate(model)
```

## Development

This project follows **readme-driven development** and **UI/UX-driven development** principles, where the user experience is defined through the standardized interfaces for prediction models and evaluation methods.

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run type checking
mypy ncaa_eval

# Format code
ruff format .
ruff check .
```

### Code Quality Standards

- All functions must have comprehensive docstrings following Google style
- Maximum of 7 logical statements per function
- DRY principle adherence
- Maximum line length of 88 characters
- Type hints required for all functions

## Contributing

See [Contributing](contributing.md) for detailed guidelines on contributing to this project.

## Authors

Dan Hilgart <dhilgart@gmail.com>

## Acknowledgments

Created from [Lee-W/cookiecutter-python-template](https://github.com/Lee-W/cookiecutter-python-template/tree/1.11.0) version 1.11.0

Evaluation methodologies and model types referenced from Kaggle's March Machine Learning Mania competitions:
- [March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025)
- [March Machine Learning Mania 2024](https://www.kaggle.com/competitions/march-machine-learning-mania-2024)
