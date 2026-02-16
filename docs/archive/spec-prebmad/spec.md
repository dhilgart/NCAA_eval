I want to create a python repo that evaluates NCAA tournament prediction models.
The repo should define a standard interface for prediction models to implement.
The repo should define a standard interface for different ways of evaluating the models.
The repo should have the ability to select which evaluation method(s) to use.
Typical evaluation methods should be included by default, but users should be able to add their own evaluation methods.
Typical evaluation methods should be scraped from https://www.kaggle.com/competitions/march-machine-learning-mania-2025, https://www.kaggle.com/competitions/march-machine-learning-mania-2024, etc.
The needs for the standard interfaces should be defined such that they support all typical model types. Typical model types can be scraped from https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion, https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion, etc.
The UX should be simple and well defined.
The repo should use ruff for code quality and formatting.
The repo should use pytest for testing.
The repo should use pydantic for data validation.
The repo should use isort for import sorting.
The repo should use pre-commit for code quality checks.
The repo should use edgetest for dependency management.
The repo should use mypy for type checking.

The repo should follow both readme-driven development and UI/UX-driven development where in this case the UI/UX is the interfaces for the prediction models and evaluation methods.