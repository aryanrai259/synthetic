# Synthetic Marketing Data Generator using GANs

This project implements a Generative Adversarial Network (GAN) to generate synthetic marketing data that preserves the statistical properties of real marketing data while ensuring privacy and providing an unlimited source of training data for marketing analytics and machine learning models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the GAN](#training-the-gan)
  - [Generating Synthetic Data](#generating-synthetic-data)
  - [Evaluating Generated Data](#evaluating-generated-data)
- [GAN Architecture](#gan-architecture)
- [Data Processing](#data-processing)
- [Evaluation Metrics](#evaluation-metrics)
- [Sample Results](#sample-results)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

Generative Adversarial Networks (GANs) are powerful deep learning models that can learn to generate synthetic data that closely resembles real data. This project applies GANs to marketing data, allowing marketers and data scientists to:

1. Generate unlimited synthetic marketing data for testing and development
2. Preserve privacy by not using real customer data
3. Augment existing datasets for better model training
4. Create balanced datasets for underrepresented segments

## Features

- **Data Processing**: Automatic handling of numerical, categorical, and binary data
- **GAN Architecture**: Customizable generator and discriminator networks
- **Training Pipeline**: Complete training workflow with checkpointing and monitoring
- **Synthetic Data Generation**: Generate any number of synthetic marketing data samples
- **Comprehensive Evaluation**: Evaluate the quality of synthetic data using multiple metrics
- **Visualization**: Visualize distributions, correlations, and embeddings of real vs. synthetic data

## Project Structure

```
synthetic_marketing_data_gan/
├── data/                      # Directory for input data files
│   └── sample_marketing_data.csv  # Sample marketing data for testing
├── models/                    # Directory for saved models
├── output/                    # Directory for generated data and evaluation results
├── src/                       # Source code
│   ├── data_processor.py      # Data preprocessing and encoding
│   ├── gan_model.py           # GAN architecture and training
│   ├── data_generator.py      # Synthetic data generation and evaluation
│   ├── train_gan.py           # Script for training the GAN
│   └── generate_samples.py    # Script for generating synthetic samples
├── requirements.txt           # Required Python packages
├── run_training.bat/sh        # Scripts to run training (Windows/Unix)
├── generate_data.bat/sh       # Scripts to generate data (Windows/Unix)
└── README.md                  # Documentation
```

## Requirements

- Python 3.7 or higher
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn
- tqdm

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd synthetic_marketing_data_gan
```

2. Create a virtual environment (recommended):
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the GAN

To train the GAN on your marketing data, use the `train_gan.py` script:

```bash
python src/train_gan.py --data_path data/your_marketing_data.csv --epochs 100 --batch_size 32 --evaluate
```

Or use the provided run scripts:

```bash
# On Windows:
run_training.bat

# On macOS/Linux:
chmod +x run_training.sh
./run_training.sh
```

#### Training Parameters

- `--data_path`: Path to your marketing data file (CSV or Excel)
- `--output_dir`: Directory to save output files (default: "../output")
- `--model_dir`: Directory to save model files (default: "../models")
- `--epochs`: Number of epochs to train for (default: 100)
- `--batch_size`: Batch size for training (default: 32)
- `--latent_dim`: Dimension of the latent space (default: 100)
- `--save_interval`: Interval (in epochs) at which to save models (default: 10)
- `--generate_samples`: Number of synthetic samples to generate after training (default: 1000)
- `--evaluate`: Evaluate the generated data after training (flag)

### Generating Synthetic Data

After training, you can generate synthetic marketing data using the `generate_samples.py` script:

```bash
python src/generate_samples.py --preprocessor_path models/preprocessor.pkl --generator_path models/generator_epoch_100.h5 --output_path output/generated_data.csv --num_samples 1000 --evaluate
```

Or use the provided run scripts:

```bash
# On Windows:
generate_data.bat

# On macOS/Linux:
chmod +x generate_data.sh
./generate_data.sh
```

#### Generation Parameters

- `--preprocessor_path`: Path to the saved preprocessor file
- `--generator_path`: Path to the saved generator model
- `--output_path`: Path to save the generated samples
- `--num_samples`: Number of samples to generate (default: 1000)
- `--latent_dim`: Dimension of the latent space (default: 100)
- `--real_data_path`: Path to real data for evaluation (optional)
- `--evaluate`: Evaluate the generated data (flag)

### Evaluating Generated Data

The evaluation is automatically performed if the `--evaluate` flag is used. The evaluation includes:

1. Distribution comparison for numerical variables
2. Correlation matrix comparison
3. Categorical distribution comparison
4. Data embedding visualization (t-SNE and PCA)
5. Silhouette score to measure separation between real and synthetic data

Results are saved in the `output/evaluation/` directory, including:
- Plots of distributions, correlations, and embeddings
- A text report with quantitative metrics
- The synthetic data in CSV format

## GAN Architecture

The GAN consists of two neural networks:

1. **Generator**: Takes random noise as input and generates synthetic marketing data
   - Input: Random noise vector (latent space)
   - Hidden layers: Dense layers with LeakyReLU activation and batch normalization
   - Output: Synthetic data with sigmoid activation

2. **Discriminator**: Tries to distinguish between real and synthetic data
   - Input: Marketing data (real or synthetic)
   - Hidden layers: Dense layers with LeakyReLU activation and dropout
   - Output: Probability that the input is real (sigmoid activation)

The networks are trained adversarially:
- The discriminator is trained to correctly classify real and synthetic data
- The generator is trained to fool the discriminator

## Data Processing

The data processor handles:

1. **Data Loading**: Supports CSV and Excel files
2. **Data Analysis**: Automatically identifies numerical, categorical, and binary columns
3. **Preprocessing**:
   - Numerical data: Min-max scaling
   - Categorical data: One-hot encoding
   - Binary data: Direct encoding
4. **Inverse Transformation**: Converts generated data back to the original format

## Evaluation Metrics

The quality of synthetic data is evaluated using:

1. **Distribution Similarity**: Compares means and standard deviations of numerical variables
2. **Correlation Preservation**: Measures how well the correlations between variables are preserved
3. **Categorical Distribution**: Uses Jensen-Shannon divergence to compare categorical distributions
4. **Embedding Separation**: Uses silhouette score to measure how distinguishable real and synthetic data are

## Sample Results

After running the evaluation, you'll find visualizations in the `output/evaluation/` directory:

- Distribution comparisons for each numerical variable
- Correlation matrix heatmaps
- Categorical distribution bar charts
- t-SNE and PCA embeddings of real and synthetic data

## Customization

### Modifying the GAN Architecture

You can customize the GAN architecture by modifying the `generator_layers` and `discriminator_layers` parameters:

```python
gan = MarketingDataGAN(
    input_dim=input_dim,
    latent_dim=100,
    generator_layers=[128, 256, 512],  # Customize layer sizes
    discriminator_layers=[512, 256, 128]  # Customize layer sizes
)
```

### Adding New Evaluation Metrics

To add new evaluation metrics, extend the `SyntheticDataGenerator` class in `data_generator.py`:

```python
def my_custom_metric(self, real_data, synthetic_data):
    # Implement your custom metric
    return metric_value
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: If you encounter memory errors during training, try:
   - Reducing the batch size
   - Using a smaller network architecture
   - Processing a subset of the data

2. **Mode Collapse**: If the generator produces limited variety, try:
   - Adjusting the learning rate
   - Adding more layers to the discriminator
   - Implementing techniques like feature matching or minibatch discrimination

3. **Poor Quality Synthetic Data**: If the synthetic data quality is poor, try:
   - Training for more epochs
   - Adjusting the network architecture
   - Ensuring proper preprocessing of the input data

## License

This project is licensed under the MIT License - see the LICENSE file for details.
