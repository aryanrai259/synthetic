@echo off
echo Training GAN for Synthetic Marketing Data Generation
echo ===================================================
echo.

python src/train_gan.py --data_path data/sample_marketing_data.csv --epochs 100 --batch_size 32 --evaluate

echo.
echo Training completed!
pause
