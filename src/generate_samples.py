"""
Generate Samples Script

This script generates synthetic marketing data samples using a trained GAN model.
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
    parser = argparse.ArgumentParser(description="Generate synthetic marketing data samples")
    
    parser.add_argument("--preprocessor_path", type=str, required=True,
                        help="Path to the saved preprocessor file")
    
    parser.add_argument("--generator_path", type=str, required=True,
                        help="Path to the saved generator model")
    
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the generated samples")
    
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to generate")
    
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="Dimension of the latent space")
    
    parser.add_argument("--real_data_path", type=str,
                        help="Path to real data for evaluation (optional)")
    
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the generated data")
    
    return parser.parse_args()


def main():
    """Main function to generate synthetic data samples."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load preprocessor
    print(f"Loading preprocessor from {args.preprocessor_path}...")
    data_processor = DataProcessor()
    data_processor.load_preprocessor(args.preprocessor_path)
    
    # Set up GAN model
    print("Setting up GAN model...")
    input_dim = data_processor.input_dim
    gan = MarketingDataGAN(input_dim=input_dim, latent_dim=args.latent_dim)
    
    # Load generator model
    print(f"Loading generator model from {args.generator_path}...")
    gan.generator = tf.keras.models.load_model(args.generator_path)
    
    # Create data generator
    data_generator = SyntheticDataGenerator(gan, data_processor)
    
    # Generate synthetic data
    print(f"Generating {args.num_samples} synthetic samples...")
    synthetic_data = data_generator.generate_data(args.num_samples)
    
    # Save synthetic data
    data_generator.save_generated_data(synthetic_data, args.output_path)
    
    # Evaluate if requested
    if args.evaluate and args.real_data_path:
        print("Evaluating synthetic data...")
        
        # Load real data
        real_data_processor = DataProcessor(args.real_data_path)
        real_data_processor.load_data()
        real_data = real_data_processor.data
        
        # Generate evaluation report
        evaluation_dir = os.path.join(os.path.dirname(args.output_path), "evaluation")
        data_generator.generate_evaluation_report(
            real_data=real_data,
            num_samples=args.num_samples,
            output_dir=evaluation_dir
        )
    
    print("Done!")


if __name__ == "__main__":
    main()
