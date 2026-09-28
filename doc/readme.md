## About

This project uses **520 survey responses** with age, gender, symptoms, and a `class` label of `Positive` or `Negative`.

The goal is to predict the **probability of a Positive class** from the other columns.

- `X` = input features
- `y` = target class

> This project is for learning only and is **not a medical diagnosis**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn
```
## Follow the steps

1. **`eda.py` — explore.** Read the original CSV, count missing and repeated rows, and draw the class and symptom charts in `doc/img/eda/`.