python stable_test.py --data_path ../raw_data/all.xlsx --output_dir ../output/control --preprocess_func default
python stable_test.py --data_path ../raw_data/all.xlsx --output_dir ../output/miNNseq --preprocess_func miNNseq

python regression_stable_test.py --data_path ../raw_data/all.xlsx --output_dir ../output/regression_control --preprocess_func default

python RF_feature_importance.py --data_path ../raw_data/all.xlsx --output_dir ../output/control --preprocess_func default
python RF_feature_importance.py --data_path ../raw_data/all.xlsx --output_dir ../output/miNNseq --preprocess_func miNNseq

python stable_test.py --data_path ../raw_data/all.xlsx --output_dir ../output/without_MHT --preprocess_func miNNseq --ignore_column "Mean hearing threshold (affected side)" --ignore_column "WHO classify (affected side)"
python stable_test_WHO.py --data_path ../raw_data/all.xlsx --after_treatment_data_path ../raw_data/after_treatment.xlsx --output_dir ../output/WHO_predict --preprocess_func miNNseq
