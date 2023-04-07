## Mini GPT

<hr>

*Hans' mini GPT codebase.*

<br>

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

`mini-gpt` is a package that contains:

- Preprocessor classes
- Model classes
- Workflow scripts
- Visualization

## Documentation

The official documentation is hosted on this repository's [wiki](), along with a long-term roadmap. A short-term todo list for this project is posted on our [kanban board]().


## Installation

### Linux

Set up a Python 3.11.3 virtual environment, then make the following local invocations from the terminal:

```
pip install -e .[linux]

pre-commit install

pre-commit autoupdate
```

## Unit tests

After installation, make the following local invocation from the terminal:
```
pytest
```

## Quick Start

Running locally:
```
python workflows/mini-gpt/data/preprocessing.py --dataSource=local

python workflows/mini-gpt/model/train.py --modelType=classification --workflowMode=dev --dataSource=local --modelVersion=main_20230214
```
