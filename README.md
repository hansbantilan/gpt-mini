## GPT-mini

<hr>

*GPT-mini codebase.*

<br>

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

`gpt-mini` is a package that contains:

- Model classes
- Workflow scripts

## Documentation

The official documentation is hosted on this repository's [wiki](), along with a long-term roadmap. A short-term todo list for this project is posted on our [kanban board]().


## Installation

### Linux

Set up a Python 3.10.10 virtual environment, then make the following local invocations from the terminal:

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
python workflows/gpt/model/train.py --modelType=tensorflow_char --modelVersion=hb_20220412 --dataSource=shakespeare
python workflows/gpt/model/score.py --modelType=tensorflow_char --modelVersion=hb_20220412 --dataSource=shakespeare
```
