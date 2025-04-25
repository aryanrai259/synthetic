"""
Data Processor Module

This module handles loading, preprocessing, and encoding marketing data for GAN training.
It provides utilities for data normalization, encoding categorical variables, and preparing
data batches for training.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle


class DataProcessor:
    """
    A class for processing marketing data for GAN training.
    """

    def __init__(self, data_path=None):
        """
        Initialize the DataProcessor.

        Args:
            data_path (str, optional): Path to the data file. Defaults to None.
        """
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.numerical_columns = []
        self.categorical_columns = []
        self.binary_columns = []
        self.column_info = {}
        self.numerical_scaler = MinMaxScaler()
        self.categorical_encoders = {}
        self.input_dim = 0

    def load_data(self, data_path=None):
        """
        Load data from a file.

        Args:
            data_path (str, optional): Path to the data file. If None, uses the path provided during initialization.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            FileNotFoundError: If the data file does not exist.
            ValueError: If the data file is empty or invalid.
        """
        if data_path:
            self.data_path = data_path

        if not self.data_path:
            raise ValueError("Data path is required.")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        file_extension = os.path.splitext(self.data_path)[1].lower()

        try:
            if file_extension == '.csv':
                self.data = pd.read_csv(self.data_path)
            elif file_extension in ['.xlsx', '.xls']:
                self.data = pd.read_excel(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            if self.data.empty:
                raise ValueError("Data file is empty")

            print(f"Successfully loaded data from {self.data_path}")
            print(f"Data shape: {self.data.shape}")
            return self.data

        except Exception as e:
            raise ValueError(f"Error loading data file: {str(e)}")

    def analyze_data(self):
        """
        Analyze the data and identify column types.

        Returns:
            dict: A dictionary containing information about the columns.
        """
        if self.data is None:
            raise ValueError("Data has not been loaded. Call load_data() first.")

        self.column_info = {}
        self.numerical_columns = []
        self.categorical_columns = []
        self.binary_columns = []

        for column in self.data.columns:
            column_data = self.data[column]
            unique_values = column_data.nunique()
            
            # Check if column is binary (has only 2 unique values)
            if unique_values == 2:
                self.binary_columns.append(column)
                self.column_info[column] = {
                    'type': 'binary',
                    'unique_values': column_data.unique().tolist()
                }
            # Check if column is categorical (has few unique values or is object type)
            elif unique_values < 10 or column_data.dtype == 'object':
                self.categorical_columns.append(column)
                self.column_info[column] = {
                    'type': 'categorical',
                    'unique_values': column_data.unique().tolist(),
                    'num_categories': unique_values
                }
            # Otherwise, treat as numerical
            else:
                self.numerical_columns.append(column)
                self.column_info[column] = {
                    'type': 'numerical',
                    'min': column_data.min(),
                    'max': column_data.max(),
                    'mean': column_data.mean(),
                    'std': column_data.std()
                }

        print(f"Numerical columns: {len(self.numerical_columns)}")
        print(f"Categorical columns: {len(self.categorical_columns)}")
        print(f"Binary columns: {len(self.binary_columns)}")

        return self.column_info

    def preprocess_data(self):
        """
        Preprocess the data by handling missing values, encoding categorical variables,
        and scaling numerical variables.

        Returns:
            np.ndarray: The preprocessed data as a numpy array.
        """
        if self.data is None:
            raise ValueError("Data has not been loaded. Call load_data() first.")

        if not self.column_info:
            self.analyze_data()

        # Handle missing values
        self.data = self.data.fillna(self.data.mean())

        # Process numerical columns
        numerical_data = self.data[self.numerical_columns].values
        if len(numerical_data) > 0:
            numerical_data = self.numerical_scaler.fit_transform(numerical_data)

        # Process categorical columns
        categorical_data_list = []
        for column in self.categorical_columns:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            column_data = self.data[column].values.reshape(-1, 1)
            encoded_data = encoder.fit_transform(column_data)
            self.categorical_encoders[column] = encoder
            categorical_data_list.append(encoded_data)

        # Process binary columns
        binary_data = self.data[self.binary_columns].values

        # Combine all processed data
        processed_data_parts = []
        
        if len(numerical_data) > 0:
            processed_data_parts.append(numerical_data)
        
        for encoded_data in categorical_data_list:
            processed_data_parts.append(encoded_data)
        
        if len(binary_data) > 0:
            processed_data_parts.append(binary_data)

        if processed_data_parts:
            self.processed_data = np.hstack(processed_data_parts)
        else:
            self.processed_data = np.array([])

        self.input_dim = self.processed_data.shape[1]
        print(f"Processed data shape: {self.processed_data.shape}")
        print(f"Input dimension for GAN: {self.input_dim}")

        return self.processed_data

    def get_train_test_split(self, test_size=0.2, random_state=42):
        """
        Split the processed data into training and testing sets.

        Args:
            test_size (float, optional): The proportion of the data to include in the test split. Defaults to 0.2.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.

        Returns:
            tuple: (X_train, X_test) - Training and testing data.
        """
        if self.processed_data is None:
            raise ValueError("Data has not been preprocessed. Call preprocess_data() first.")

        X_train, X_test = train_test_split(
            self.processed_data, test_size=test_size, random_state=random_state
        )

        return X_train, X_test

    def generate_batches(self, data, batch_size):
        """
        Generate batches of data for training.

        Args:
            data (np.ndarray): The data to generate batches from.
            batch_size (int): The size of each batch.

        Yields:
            np.ndarray: A batch of data.
        """
        num_samples = data.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield data[batch_indices]

    def inverse_transform(self, generated_data):
        """
        Transform generated data back to the original format.

        Args:
            generated_data (np.ndarray): The generated data.

        Returns:
            pd.DataFrame: The transformed data as a DataFrame.
        """
        if not self.column_info:
            raise ValueError("Column information is not available. Call analyze_data() first.")

        # Initialize the result DataFrame
        result_df = pd.DataFrame()
        
        # Keep track of the current column index in the generated data
        current_idx = 0
        
        # Process numerical columns
        if self.numerical_columns:
            num_numerical = len(self.numerical_columns)
            numerical_data = generated_data[:, current_idx:current_idx + num_numerical]
            numerical_data = self.numerical_scaler.inverse_transform(numerical_data)
            
            for i, column in enumerate(self.numerical_columns):
                result_df[column] = numerical_data[:, i]
            
            current_idx += num_numerical
        
        # Process categorical columns
        for column in self.categorical_columns:
            encoder = self.categorical_encoders[column]
            num_categories = encoder.categories_[0].shape[0]
            
            categorical_data = generated_data[:, current_idx:current_idx + num_categories]
            # Convert to one-hot format (ensure each row has exactly one 1)
            categorical_indices = np.argmax(categorical_data, axis=1)
            categorical_one_hot = np.zeros_like(categorical_data)
            categorical_one_hot[np.arange(categorical_data.shape[0]), categorical_indices] = 1
            
            # Inverse transform
            inverse_data = encoder.inverse_transform(categorical_one_hot)
            result_df[column] = inverse_data.flatten()
            
            current_idx += num_categories
        
        # Process binary columns
        for column in self.binary_columns:
            binary_data = generated_data[:, current_idx]
            # Round to 0 or 1
            binary_data = np.round(binary_data).astype(int)
            
            # Map back to original values if they're not 0 and 1
            unique_values = self.column_info[column]['unique_values']
            if set(unique_values) != {0, 1}:
                mapping = {0: unique_values[0], 1: unique_values[1]}
                binary_data = np.vectorize(lambda x: mapping[x])(binary_data)
            
            result_df[column] = binary_data
            current_idx += 1
        
        return result_df

    def save_preprocessor(self, file_path):
        """
        Save the preprocessor to a file.

        Args:
            file_path (str): Path to save the preprocessor.
        """
        preprocessor_data = {
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'binary_columns': self.binary_columns,
            'column_info': self.column_info,
            'numerical_scaler': self.numerical_scaler,
            'categorical_encoders': self.categorical_encoders,
            'input_dim': self.input_dim
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        print(f"Preprocessor saved to {file_path}")

    def load_preprocessor(self, file_path):
        """
        Load a preprocessor from a file.

        Args:
            file_path (str): Path to the preprocessor file.
        """
        with open(file_path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.numerical_columns = preprocessor_data['numerical_columns']
        self.categorical_columns = preprocessor_data['categorical_columns']
        self.binary_columns = preprocessor_data['binary_columns']
        self.column_info = preprocessor_data['column_info']
        self.numerical_scaler = preprocessor_data['numerical_scaler']
        self.categorical_encoders = preprocessor_data['categorical_encoders']
        self.input_dim = preprocessor_data['input_dim']
        
        print(f"Preprocessor loaded from {file_path}")
        print(f"Input dimension for GAN: {self.input_dim}")


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor("../data/marketing_data.csv")
    processor.load_data()
    processor.analyze_data()
    processed_data = processor.preprocess_data()
    X_train, X_test = processor.get_train_test_split()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Save the preprocessor
    processor.save_preprocessor("../models/preprocessor.pkl")
