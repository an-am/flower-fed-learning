"""flower-fed-learning: A Flower / TensorFlow app."""

import os

import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from datasets import Dataset, load_dataset
from numpy import ndarray

from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

INPUT_DIM = 9
NUM_CLASSES = 1
BATCH_SIZE = 64
EPOCHS = 10


def load_model():
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = Sequential()

    # Input layer
    model.add(Dense(9, activation='relu', input_shape=(INPUT_DIM,)))

    # Hidden layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    # Output layer (no activation for regression)
    model.add(Dense(1, activation='linear'))

    # Compile model with MSE loss for regression
    model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['mae'])

    return model

def data_prep(data) -> ndarray:
    # Transform the dataset into a DataFrame
    data_df = pd.DataFrame(data)

    # Compute the FinancialStatus
    data_df['FinancialStatus'] = data_df['FinancialEducation'] * np.log(data_df['Wealth'])

    # Drop ID column
    data_df = data_df.drop('ID', axis=1)

    # Drop the target column
    x = data_df.drop(columns=['RiskPropensity'])

    # BoxCox transformation on Wealth and Income
    x['Wealth'], fitted_lambda_wealth = boxcox(x['Wealth'])
    x['Income'], fitted_lambda_income = boxcox(x['Income'])

    # Transform the df to ndarray
    x = x.to_numpy()

    # Scale the data
    x_scaled = (x - x.mean()) / x.std()

    return x_scaled

def load_data(partition_id, num_partitions):
    file_path = "/Users/antonelloamore/PycharmProjects/ray-fed-training/Needs.csv"

    # Create a Hugging Face Dataset from the CSV
    dataset = load_dataset("csv", data_files=file_path)

    # Partition the data
    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = dataset["train"]
    partition = partitioner.load_partition(partition_id=partition_id)

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)

    # Do data prep + create train and test sets
    x_train = data_prep(partition["train"])
    y_train = np.asarray(partition["train"]["RiskPropensity"])

    x_test = data_prep(partition["test"])
    y_test = np.asarray(partition["test"]["RiskPropensity"])

    #print(f"x_train: {x_train},\n y_train: {y_train},\n x_test: {x_test},\n y_test: {y_test}\n")

    return x_train, y_train, x_test, y_test
