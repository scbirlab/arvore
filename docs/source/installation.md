# Installation

Python 3.11 or later is required.

```bash
pip install arvore
```

Model backends and feature tools are optional:

```bash
pip install "arvore[trees]"  # XGBoost and LightGBM
pip install "arvore[cat]"    # CatBoost
pip install "arvore[chem]"   # Molecular fingerprints and descriptors
pip install "arvore[bio]"    # Biological feature tools
```

## From source

```bash
git clone https://github.com/scbirlab/arvore.git
cd arvore
pip install -e ".[dev]"
```
