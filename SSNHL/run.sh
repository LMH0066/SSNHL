python stable_test.py --data_path ../raw_data/all.xlsx --output_dir ../output/control --preprocess_func default
python stable_test.py --data_path ../raw_data/all.xlsx --output_dir ../output/data_fill/KNN --preprocess_func KNN
python stable_test.py --data_path ../raw_data/all.xlsx --output_dir ../output/data_fill/rpca --preprocess_func rpca

python RF_feature_importance.py --data_path ../raw_data/all.xlsx --output_dir ../output/control --preprocess_func default
python RF_feature_importance.py --data_path ../raw_data/all.xlsx --output_dir ../output/data_fill/KNN --preprocess_func KNN
