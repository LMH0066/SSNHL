### develop environment prepare
```
conda create -n SSNHL python=3.9
conda activate SSNHL
pip install poetry
poetry install
```

### execute instruction
```
python stable_test.py --data_path ../raw_data/all.xlsx --output_dir ../output/control --preprocess_func default
python stable_test.py --data_path ../raw_data/all.xlsx --output_dir ../output/data_fill/KNN --preprocess_func KNN
python stable_test.py --data_path ../raw_data/all.xlsx --output_dir ../output/data_fill/rpca --preprocess_func rpca
```
