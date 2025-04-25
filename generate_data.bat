@echo off
echo Generating Synthetic Marketing Data
echo =================================
echo.

python src/generate_samples.py --preprocessor_path models/preprocessor.pkl --generator_path models/generator_epoch_100.h5 --output_path output/generated_data.csv --num_samples 1000 --real_data_path data/sample_marketing_data.csv --evaluate

echo.
echo Data generation completed!
pause
