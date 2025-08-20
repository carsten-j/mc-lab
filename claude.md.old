# CLAUDE.MD

*Last updated 2025-08-17*

1. Project Overview
MC-LAB is an educational repository for learning about various Monte Carlo statistical methods by implementing them in Python. Key components:

* src/mc_lab: Core algorithms and functionality
* notesbooks: Jupyter Notebooks demonstrating algorithms from src/mc_lab
* tests: test cases written using PyTest
  
2. Non-negotiable golden rules

   * Always asks before adding a new Python package.
   * asdg

3. Build, test & utility commands

Use Make tasks for consistency (they ensure correct environment variables and configuration).

## Format, lint, type-check, test, codegen

* make format-fix    # ruff format
* make lint-fix      # ruff check
* make test          # run test cases
* make perftest      # Run specific tests by Ward pattern*

4. Coding standards

Python: 3.12+
Formatting: ruff enforces 96-char lines, double quotes, sorted imports. Standard ruff linter rules.
Typing: Strict (Pydantic v2 models preferred); from __future__ import annotations.
Naming: snake_case (functions/variables), PascalCase (classes), SCREAMING_SNAKE (constants).
Error Handling: Typed exceptions; context managers for resources.
Documentation: Google-style docstrings for public functions/classes.
Testing: Separate test files matching source file patterns.
Error handling patterns:

Use typed, hierarchical exceptions defined in exceptions.py
Catch specific exceptions, not general Exception

5. Rules for Jupyter Notebooks

