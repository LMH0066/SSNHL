# Develop Guidelines
## environment prepare
```
conda create -n SSNHL python=3.9
conda activate SSNHL
pip install poetry
poetry install
```
## pre commit
```
# Format python code
poe format
# Run test code
poe test
```
