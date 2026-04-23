# Contributing to Cardiovascular Risk Assessment

Thank you for your interest in contributing to this project! We welcome contributions from the community, whether it's bug fixes, new features, documentation improvements, or model enhancements.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

This project adheres to our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create a branch** for your feature or fix
4. **Make changes** and test thoroughly
5. **Submit** a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/CARDIOVASCULAR-RISK-ASSESSMENT.git
cd CARDIOVASCULAR-RISK-ASSESSMENT

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## How to Contribute

### Bug Fixes
- Search existing issues first to avoid duplicates
- Create a new issue describing the bug
- Reference the issue in your pull request

### New Features
- Open an issue to discuss the proposed feature
- Wait for approval before beginning work
- Follow the existing code patterns and style

### Documentation
- Fix typos, improve clarity, or add examples
- Keep language clear and concise
- Use markdown formatting consistently

### Model Improvements
- Document any changes to model architecture or hyperparameters
- Include before/after accuracy metrics
- Ensure SHAP explainability is maintained

## Pull Request Process

1. Update the README.md if your change affects documentation
2. Ensure your code follows the project's style guidelines
3. Test your changes locally before submitting
4. Write clear, descriptive commit messages
5. Reference any related issues in your PR description

### Commit Message Format

```
type(scope): brief description

[optional body]
[optional footer]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples:**
```
feat(model): add random forest classifier option
fix(api): handle missing patient data gracefully
docs(readme): update installation instructions
```

## Style Guidelines

### Python
- Follow PEP 8 conventions
- Use docstrings for all functions and classes
- Keep functions focused and under 50 lines when possible
- Use type hints where appropriate

### HTML/CSS
- Use semantic HTML5 elements
- Maintain the professional medical theme
- Ensure responsive design for all screen sizes

### JavaScript
- Use vanilla JavaScript (no frameworks)
- Comment complex logic
- Handle errors gracefully

## Reporting Issues

When reporting issues, please include:

1. **Description** — What happened vs. what you expected
2. **Steps to reproduce** — Detailed steps to trigger the issue
3. **Environment** — OS, Python version, browser
4. **Screenshots** — If applicable
5. **Error logs** — Console or terminal output

---

Thank you for contributing!
