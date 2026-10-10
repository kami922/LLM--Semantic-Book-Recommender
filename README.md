# LLM--Semantic-Book-Recommender

A minimal semantic book recommender inspired by the linked course project. It supports:

- Natural-language queries against book descriptions
- Semantic similarity ranking using TF-IDF cosine similarity
- Optional category and emotion filtering

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python app.py --query "astronaut survival on mars"
python app.py --query "build better habits" --category Nonfiction
```

## Test

```bash
python -m unittest discover -s tests
```
