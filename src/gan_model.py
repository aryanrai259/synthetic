"""
GAN Model Module

This module implements the Generative Adversarial Network (GAN) architecture for
generating synthetic marketing data. It includes the Generator and Discriminator
models, as well as the GAN training loop.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


class MarketingDataGAN:
    """
    A class implementing a GAN for generating synthetic marketing data.
    """

    def __init__(self, input_dim, latent_dim=100, generator_layers=[128, 256], discriminator_layers=[256, 128]):
        """
        Initialize the GAN model.

        Args:
            input_dim (int): Dimension of the input data (after preprocessing).
            latent_dim (int, optional): Dimension of the latent space. Defaults to 100.
            generator_layers (list, optional): List of layer sizes for the generator. Defaults to [128, 256].
            discriminator_layers (list, optional): List of layer sizes for the discriminator. Defaults to [256, 128].
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator_layers = generator_layers
        self.discriminator_layers = discriminator_layers
        
        # Build the generator and discriminator
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Define optimizers
        self.generator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Metrics
        self.gen_loss_metric = tf.keras.metrics.Mean(name='generator_loss')
        self.disc_loss_metric = tf.keras.metrics.Mean(name='discriminator_loss')
        
        # Training history
        self.history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'epochs': []
        }

    def _build_generator(self):
        """
        Build the generator model.

        Returns:
            tf.keras.Model: The generator model.
        """
        model = models.Sequential(name='generator')
        
        # First layer
        model.add(layers.Dense(self.generator_layers[0], input_dim=self.latent_dim))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization())
        
        # Hidden layers
        for layer_size in self.generator_layers[1:]:
            model.add(layers.Dense(layer_size))
            model.add(layers.LeakyReLU(alpha=0.2))
            model.add(layers.BatchNormalization())
        
        # Output layer
        model.add(layers.Dense(self.input_dim, activation='sigmoid'))
        
        model.summary()
        return model

    def _build_discriminator(self):
        """
        Build the discriminator model.

        Returns:
            tf.keras.Model: The discriminator model.
        """
        model = models.Sequential(name='discriminator')
        
        # First layer
        model.add(layers.Dense(self.discriminator_layers[0], input_dim=self.input_dim))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        
        # Hidden layers
        for layer_size in self.discriminator_layers[1:]:
            model.add(layers.Dense(layer_size))
            model.add(layers.LeakyReLU(alpha=0.2))
            model.add(layers.Dropout(0.3))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.summary()
        return model

    @tf.function
    def _train_discriminator_step(self, real_data, batch_size):
        """
        Perform one training step for the discriminator.

        Args:
            real_data (tf.Tensor): Batch of real data.
            batch_size (int): Size of the batch.

        Returns:
            float: Discriminator loss.
        """
        # Generate random noise
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as disc_tape:
            # Generate fake data
            generated_data = self.generator(noise, training=True)
            
            # Get discriminator outputs for real and fake data
            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator(generated_data, training=True)
            
            # Calculate losses
            real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(
                tf.ones_like(real_output), real_output
            )
            fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(
                tf.zeros_like(fake_output), fake_output
            )
            disc_loss = real_loss + fake_loss
        
        # Calculate gradients and update discriminator
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        
        return disc_loss

    @tf.function
    def _train_generator_step(self, batch_size):
        """
        Perform one training step for the generator.

        Args:
            batch_size (int): Size of the batch.

        Returns:
            float: Generator loss.
        """
        # Generate random noise
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape:
            # Generate fake data
            generated_data = self.generator(noise, training=True)
            
            # Get discriminator output for fake data
            fake_output = self.discriminator(generated_data, training=True)
            
            # Calculate loss (we want the discriminator to think these are real)
            gen_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(
                tf.ones_like(fake_output), fake_output
            )
        
        # Calculate gradients and update generator
        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        
        return gen_loss

    def train(self, data, epochs=100, batch_size=32, save_interval=10, save_dir='../models'):
        """
        Train the GAN model.

        Args:
            data (np.ndarray): Training data.
            epochs (int, optional): Number of epochs to train for. Defaults to 100.
            batch_size (int, optional): Batch size. Defaults to 32.
            save_interval (int, optional): Interval (in epochs) at which to save models. Defaults to 10.
            save_dir (str, optional): Directory to save models. Defaults to '../models'.

        Returns:
            dict: Training history.
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert data to tensor
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        
        # Calculate number of batches
        num_samples = data.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Reset metrics
            self.gen_loss_metric.reset_states()
            self.disc_loss_metric.reset_states()
            
            # Shuffle data
            indices = tf.range(start=0, limit=num_samples, dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            shuffled_data = tf.gather(data, shuffled_indices)
            
            # Train on batches
            for batch in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}"):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, num_samples)
                batch_data = shuffled_data[start_idx:end_idx]
                
                # Train discriminator
                disc_loss = self._train_discriminator_step(batch_data, end_idx - start_idx)
                self.disc_loss_metric.update_state(disc_loss)
                
                # Train generator
                gen_loss = self._train_generator_step(end_idx - start_idx)
                self.gen_loss_metric.update_state(gen_loss)
            
            # Record losses
            epoch_gen_loss = self.gen_loss_metric.result()
            epoch_disc_loss = self.disc_loss_metric.result()
            
            self.history['generator_loss'].append(float(epoch_gen_loss))
            self.history['discriminator_loss'].append(float(epoch_disc_loss))
            self.history['epochs'].append(epoch + 1)
            
            # Print progress
            time_taken = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Generator Loss: {epoch_gen_loss:.4f}, "
                  f"Discriminator Loss: {epoch_disc_loss:.4f}, "
                  f"Time: {time_taken:.2f}s")
            
            # Save models at specified intervals
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
                self.save_models(save_dir, epoch + 1)
                self.plot_losses(save_dir)
        
        return self.history

    def generate_samples(self, num_samples):
        """
        Generate synthetic samples.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: Generated samples.
        """
        # Generate random noise
        noise = tf.random.normal([num_samples, self.latent_dim])
        
        # Generate samples
        generated_samples = self.generator(noise, training=False)
        
        return generated_samples.numpy()

    def save_models(self, save_dir, epoch):
        """
        Save the generator and discriminator models.

        Args:
            save_dir (str): Directory to save models.
            epoch (int): Current epoch number.
        """
        generator_path = os.path.join(save_dir, f"generator_epoch_{epoch}.h5")
        discriminator_path = os.path.join(save_dir, f"discriminator_epoch_{epoch}.h5")
        
        self.generator.save(generator_path)
        self.discriminator.save(discriminator_path)
        
        print(f"Models saved at epoch {epoch}")

    def load_models(self, generator_path, discriminator_path):
        """
        Load the generator and discriminator models.

        Args:
            generator_path (str): Path to the generator model.
            discriminator_path (str): Path to the discriminator model.
        """
        self.generator = models.load_model(generator_path)
        self.discriminator = models.load_model(discriminator_path)
        
        print("Models loaded successfully")

    def plot_losses(self, save_dir):
        """
        Plot the training losses.

        Args:
            save_dir (str): Directory to save the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epochs'], self.history['generator_loss'], label='Generator Loss')
        plt.plot(self.history['epochs'], self.history['discriminator_loss'], label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Losses')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(save_dir, 'loss_plot.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Loss plot saved to {plot_path}")


if __name__ == "__main__":
    # Example usage
    input_dim = 20  # Example dimension
    gan = MarketingDataGAN(input_dim)
    
    # Generate some random data for testing
    test_data = np.random.rand(1000, input_dim)
    
    # Train for a few epochs
    gan.train(test_data, epochs=5, batch_size=32)
    
    # Generate samples
    samples = gan.generate_samples(10)
    print(f"Generated samples shape: {samples.shape}")
