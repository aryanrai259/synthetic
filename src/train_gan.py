"""
GAN Training Script

This script trains a GAN model to generate synthetic marketing data.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from data_processor import DataProcessor
from gan_model import MarketingDataGAN
from data_generator import SyntheticDataGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a GAN to generate synthetic marketing data")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the marketing data file (CSV or Excel)")
    
    parser.add_argument("--output_dir", type=str, default="../output",
                        help="Directory to save output files")
    
    parser.add_argument("--model_dir", type=str, default="../models",
                        help="Directory to save model files")
    
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train for")
    
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="Dimension of the latent space")
    
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Interval (in epochs) at which to save models")
    
    parser.add_argument("--generate_samples", type=int, default=1000,
                        help="Number of synthetic samples to generate after training")
    
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the generated data after training")
    
    return parser.parse_args()


def main():
    """Main function to train the GAN and generate synthetic data."""
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Set up data processor
    print(f"Loading data from {args.data_path}...")
    data_processor = DataProcessor(args.data_path)
    data_processor.load_data()
    
    # Analyze and preprocess data
    print("Analyzing data...")
    data_processor.analyze_data()
    
    print("Preprocessing data...")
    processed_data = data_processor.preprocess_data()

    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available:", len(gpus))
    for gpu in gpus:
        print("GPU Name:", gpu.name)

    
    # Save the preprocessor
    preprocessor_path = os.path.join(args.model_dir, "preprocessor.pkl")
    data_processor.save_preprocessor(preprocessor_path)
    
    # Split data
    print("Splitting data into training and testing sets...")
    X_train, X_test = data_processor.get_train_test_split()
    
    # Set up GAN model
    print("Building GAN model...")
    input_dim = data_processor.input_dim
    gan = MarketingDataGAN(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        generator_layers=[128, 256, 512],
        discriminator_layers=[512, 256, 128]
    )
    
    # Train GAN
    print(f"Training GAN for {args.epochs} epochs...")
    with tf.device('/GPU:0'):
        gan.train(
            data=X_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            save_dir=args.model_dir
        )
    
    # Generate synthetic data
    print(f"Generating {args.generate_samples} synthetic samples...")
    data_generator = SyntheticDataGenerator(gan, data_processor)
    synthetic_data = data_generator.generate_data(args.generate_samples)
    
    # Save synthetic data
    synthetic_data_path = os.path.join(args.output_dir, "synthetic_data.csv")
    data_generator.save_generated_data(synthetic_data, synthetic_data_path)
    
    # Evaluate if requested
    if args.evaluate:
        print("Evaluating synthetic data...")
        evaluation_dir = os.path.join(args.output_dir, "evaluation")
        data_generator.generate_evaluation_report(
            real_data=data_processor.data,
            num_samples=args.generate_samples,
            output_dir=evaluation_dir
        )
    
    print("Done!")


if __name__ == "__main__":
    main()
