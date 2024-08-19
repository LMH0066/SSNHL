python stable_test.py --data_path ../raw_data/all.xlsx --output_dir ../output/control --preprocess_func default
python stable_test.py --data_path ../raw_data/all.xlsx --output_dir ../output/miNNseq --preprocess_func miNNseq

python regression_stable_test.py --data_path ../raw_data/all.xlsx --output_dir ../output/regression_control --preprocess_func default

python RF_feature_importance.py --data_path ../raw_data/all.xlsx --output_dir ../output/control --preprocess_func default
python RF_feature_importance.py --data_path ../raw_data/all.xlsx --output_dir ../output/miNNseq --preprocess_func miNNseq
