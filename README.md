# Data Accuracy Score (DAS)

**DAS** measures how faithfully **synthetic data** preserves the predictive structure of the **original data**.

- **Classification DAS**: % agreement between predictions from a model trained on original vs. synthetic data, evaluated on the same original test set.  
- **Regression DAS**: Similarity between Original data and Synthetic Data's R² score.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)

---

## Why DAS?

Synthetic data is useful only if models trained on it behave like models trained on the original data. DAS offers a simple, model-agnostic check using any estimator with `fit` and `predict`.

---

## Install (local clone)

```bash
git clone https://github.com/pragatischdv/data-accuracy-score.git
cd data-accuracy-score
pip install -r requirements.txt
```

> Colab quick test (no install):  
> ```python
> !git clone https://github.com/pragatischdv/data-accuracy-score.git
> %cd data-accuracy-score
> from das import evaluate_das
> ```

---

## Minimal API

```python
from das import evaluate_das
```

- Pass **unfitted** estimator instances (any sklearn-like model exposing `fit` and `predict`).
- Inputs can be NumPy arrays or pandas DataFrames (columns must align across splits).

---

## Project Structure

```
das/
  __init__.py
  api.py               
  metrics.py           
  validation.py       
tests/
  DAS_Normalizing_Flows.ipynb
  DAS_test.ipynb
```
---

## Citation

If you use this repository or the DAS metric, please cite:

> Pragati Sachdeva and Amarjit Malhotra.  
> **“Evaluating Normalizing Flow Model Variants for Supervised Tabular Data: A Comparative Analysis Using the Data Accuracy Score.”**  
> *Procedia Computer Science*, 2025. doi: **10.1016/j.procs.2025.04.271**

You can also use the **CITATION.cff** included in the repo.

---
