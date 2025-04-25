"""
Data Generator Module

This module provides utilities for generating synthetic marketing data using a trained GAN model
and evaluating the quality of the generated data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import tensorflow as tf


class SyntheticDataGenerator:
    """
    A class for generating and evaluating synthetic marketing data.
    """

    def __init__(self, gan_model, data_processor):
        """
        Initialize the SyntheticDataGenerator.

        Args:
            gan_model (MarketingDataGAN): Trained GAN model.
            data_processor (DataProcessor): Data processor used for preprocessing and inverse transformation.
        """
        self.gan_model = gan_model
        self.data_processor = data_processor

    def generate_data(self, num_samples):
        """
        Generate synthetic marketing data.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            pd.DataFrame: Generated data as a DataFrame.
        """
        # Generate raw samples
        raw_samples = self.gan_model.generate_samples(num_samples)
        
        # Transform to original format
        generated_df = self.data_processor.inverse_transform(raw_samples)
        
        return generated_df

    def save_generated_data(self, data, file_path):
        """
        Save generated data to a file.

        Args:
            data (pd.DataFrame): Generated data.
            file_path (str): Path to save the data.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Determine file format based on extension
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            data.to_csv(file_path, index=False)
        elif file_extension in ['.xlsx', '.xls']:
            data.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        print(f"Generated data saved to {file_path}")

    def evaluate_distribution(self, real_data, synthetic_data, numerical_columns, output_dir):
        """
        Evaluate the distribution of synthetic data compared to real data.

        Args:
            real_data (pd.DataFrame): Real data.
            synthetic_data (pd.DataFrame): Synthetic data.
            numerical_columns (list): List of numerical columns to evaluate.
            output_dir (str): Directory to save the plots.

        Returns:
            dict: Evaluation metrics.
        """
        os.makedirs(output_dir, exist_ok=True)
        metrics = {}
        
        # Histogram comparison for numerical columns
        for column in numerical_columns:
            if column in real_data.columns and column in synthetic_data.columns:
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                sns.histplot(real_data[column], kde=True)
                plt.title(f'Real Data - {column}')
                
                plt.subplot(1, 2, 2)
                sns.histplot(synthetic_data[column], kde=True)
                plt.title(f'Synthetic Data - {column}')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'distribution_{column}.png'))
                plt.close()
                
                # Calculate distribution similarity metrics
                real_mean = real_data[column].mean()
                synth_mean = synthetic_data[column].mean()
                real_std = real_data[column].std()
                synth_std = synthetic_data[column].std()
                
                mean_diff = abs(real_mean - synth_mean) / max(abs(real_mean), 1e-10)
                std_diff = abs(real_std - synth_std) / max(abs(real_std), 1e-10)
                
                metrics[column] = {
                    'mean_difference_pct': mean_diff * 100,
                    'std_difference_pct': std_diff * 100
                }
        
        return metrics

    def evaluate_correlations(self, real_data, synthetic_data, numerical_columns, output_dir):
        """
        Evaluate the correlations in synthetic data compared to real data.

        Args:
            real_data (pd.DataFrame): Real data.
            synthetic_data (pd.DataFrame): Synthetic data.
            numerical_columns (list): List of numerical columns to evaluate.
            output_dir (str): Directory to save the plots.

        Returns:
            float: Correlation matrix difference.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter data to include only numerical columns
        real_numerical = real_data[numerical_columns]
        synth_numerical = synthetic_data[numerical_columns]
        
        # Calculate correlation matrices
        real_corr = real_numerical.corr()
        synth_corr = synth_numerical.corr()
        
        # Plot correlation matrices
        plt.figure(figsize=(20, 8))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(real_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Real Data Correlation Matrix')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(synth_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Synthetic Data Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_comparison.png'))
        plt.close()
        
        # Calculate correlation matrix difference
        corr_diff = np.abs(real_corr - synth_corr).mean().mean()
        
        return corr_diff

    def visualize_data_embedding(self, real_data, synthetic_data, output_dir, method='tsne'):
        """
        Visualize the embedding of real and synthetic data.

        Args:
            real_data (pd.DataFrame): Real data.
            synthetic_data (pd.DataFrame): Synthetic data.
            output_dir (str): Directory to save the plots.
            method (str, optional): Embedding method ('tsne' or 'pca'). Defaults to 'tsne'.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data
        real_processed = self.data_processor.preprocess_data(real_data.copy())
        synth_processed = self.data_processor.preprocess_data(synthetic_data.copy())
        
        # Combine data for embedding
        combined_data = np.vstack([real_processed, synth_processed])
        labels = np.array(['Real'] * len(real_processed) + ['Synthetic'] * len(synth_processed))
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            embedding = TSNE(n_components=2, random_state=42).fit_transform(combined_data)
            title = 't-SNE Embedding of Real and Synthetic Data'
        else:  # PCA
            embedding = PCA(n_components=2, random_state=42).fit_transform(combined_data)
            title = 'PCA Embedding of Real and Synthetic Data'
        
        # Create DataFrame for plotting
        embedding_df = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'type': labels
        })
        
        # Plot embedding
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=embedding_df, x='x', y='y', hue='type', alpha=0.7)
        plt.title(title)
        plt.savefig(os.path.join(output_dir, f'{method}_embedding.png'))
        plt.close()
        
        # Calculate silhouette score to measure separation
        silhouette = silhouette_score(embedding, labels == 'Real')
        print(f"Silhouette score: {silhouette:.4f}")
        
        return silhouette

    def evaluate_categorical_distributions(self, real_data, synthetic_data, categorical_columns, output_dir):
        """
        Evaluate the distribution of categorical variables.

        Args:
            real_data (pd.DataFrame): Real data.
            synthetic_data (pd.DataFrame): Synthetic data.
            categorical_columns (list): List of categorical columns to evaluate.
            output_dir (str): Directory to save the plots.

        Returns:
            dict: Evaluation metrics for categorical columns.
        """
        os.makedirs(output_dir, exist_ok=True)
        metrics = {}
        
        for column in categorical_columns:
            if column in real_data.columns and column in synthetic_data.columns:
                # Calculate value counts
                real_counts = real_data[column].value_counts(normalize=True)
                synth_counts = synthetic_data[column].value_counts(normalize=True)
                
                # Align indices
                all_categories = sorted(set(real_counts.index) | set(synth_counts.index))
                real_counts = real_counts.reindex(all_categories, fill_value=0)
                synth_counts = synth_counts.reindex(all_categories, fill_value=0)
                
                # Plot comparison
                plt.figure(figsize=(12, 6))
                
                # Create a DataFrame for plotting
                plot_df = pd.DataFrame({
                    'Real': real_counts,
                    'Synthetic': synth_counts
                })
                
                plot_df.plot(kind='bar')
                plt.title(f'Category Distribution - {column}')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'categorical_{column}.png'))
                plt.close()
                
                # Calculate Jensen-Shannon divergence
                js_divergence = self._jensen_shannon_divergence(real_counts.values, synth_counts.values)
                metrics[column] = {
                    'jensen_shannon_divergence': js_divergence
                }
        
        return metrics

    def _jensen_shannon_divergence(self, p, q):
        """
        Calculate Jensen-Shannon divergence between two probability distributions.

        Args:
            p (np.ndarray): First probability distribution.
            q (np.ndarray): Second probability distribution.

        Returns:
            float: Jensen-Shannon divergence.
        """
        # Ensure valid probability distributions
        p = np.asarray(p)
        q = np.asarray(q)
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate the average distribution
        m = (p + q) / 2
        
        # Calculate KL divergences
        kl_p_m = np.sum(p * np.log2(p / m, where=(p != 0)))
        kl_q_m = np.sum(q * np.log2(q / m, where=(q != 0)))
        
        # Calculate JS divergence
        js = (kl_p_m + kl_q_m) / 2
        
        return js

    def generate_evaluation_report(self, real_data, num_samples, output_dir):
        """
        Generate a comprehensive evaluation report for synthetic data.

        Args:
            real_data (pd.DataFrame): Real data.
            num_samples (int): Number of synthetic samples to generate.
            output_dir (str): Directory to save the report and plots.

        Returns:
            dict: Evaluation metrics.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate synthetic data
        synthetic_data = self.generate_data(num_samples)
        
        # Save synthetic data
        self.save_generated_data(synthetic_data, os.path.join(output_dir, 'synthetic_data.csv'))
        
        # Get column types
        numerical_columns = self.data_processor.numerical_columns
        categorical_columns = self.data_processor.categorical_columns + self.data_processor.binary_columns
        
        # Evaluate distributions
        distribution_metrics = self.evaluate_distribution(
            real_data, synthetic_data, numerical_columns, os.path.join(output_dir, 'distributions')
        )
        
        # Evaluate correlations
        correlation_diff = self.evaluate_correlations(
            real_data, synthetic_data, numerical_columns, os.path.join(output_dir, 'correlations')
        )
        
        # Evaluate categorical distributions
        categorical_metrics = self.evaluate_categorical_distributions(
            real_data, synthetic_data, categorical_columns, os.path.join(output_dir, 'categorical')
        )
        
        # Visualize data embedding
        silhouette_tsne = self.visualize_data_embedding(
            real_data, synthetic_data, os.path.join(output_dir, 'embeddings'), method='tsne'
        )
        
        silhouette_pca = self.visualize_data_embedding(
            real_data, synthetic_data, os.path.join(output_dir, 'embeddings'), method='pca'
        )
        
        # Compile report
        report = {
            'num_samples': num_samples,
            'numerical_distribution_metrics': distribution_metrics,
            'correlation_difference': correlation_diff,
            'categorical_metrics': categorical_metrics,
            'silhouette_score_tsne': silhouette_tsne,
            'silhouette_score_pca': silhouette_pca
        }
        
        # Save report
        with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
            f.write("Synthetic Marketing Data Evaluation Report\n")
            f.write("=========================================\n\n")
            f.write(f"Number of synthetic samples: {num_samples}\n\n")
            
            f.write("Numerical Distribution Metrics:\n")
            for column, metrics in distribution_metrics.items():
                f.write(f"  {column}:\n")
                for metric, value in metrics.items():
                    f.write(f"    {metric}: {value:.2f}%\n")
            
            f.write(f"\nCorrelation Matrix Difference: {correlation_diff:.4f}\n\n")
            
            f.write("Categorical Distribution Metrics:\n")
            for column, metrics in categorical_metrics.items():
                f.write(f"  {column}:\n")
                for metric, value in metrics.items():
                    f.write(f"    {metric}: {value:.4f}\n")
            
            f.write(f"\nSilhouette Score (t-SNE): {silhouette_tsne:.4f}\n")
            f.write(f"Silhouette Score (PCA): {silhouette_pca:.4f}\n")
        
        print(f"Evaluation report saved to {os.path.join(output_dir, 'evaluation_report.txt')}")
        
        return report


if __name__ == "__main__":
    # This would be used after training a GAN model
    from data_processor import DataProcessor
    from gan_model import MarketingDataGAN
    
    # Example usage
    data_processor = DataProcessor("../data/marketing_data.csv")
    data_processor.load_data()
    data_processor.analyze_data()
    processed_data = data_processor.preprocess_data()
    
    # Load a trained GAN model
    input_dim = data_processor.input_dim
    gan = MarketingDataGAN(input_dim)
    gan.load_models("../models/generator_epoch_100.h5", "../models/discriminator_epoch_100.h5")
    
    # Create data generator
    data_generator = SyntheticDataGenerator(gan, data_processor)
    
    # Generate and evaluate synthetic data
    real_data = data_processor.data
    data_generator.generate_evaluation_report(real_data, 1000, "../output/evaluation")
