import os
import scipy.io
import numpy as np
import torch
import pandas as pd

IQ_NORMALIZATION_FACTOR = 3.5

def load_test_dataset(test_df):
    tests = []
    ground_truths = []

    for index, row in test_df.iterrows():
        test_file_path = row["file_path"]
        ground_truth_file_path = row["ground_truth_file_path"]
        # print(f"test_file_path: {test_file_path}")
        # print(f"ground_truth_file_path: {ground_truth_file_path}")
        
        test = scipy.io.loadmat(test_file_path)
        NI_IQ = test['noiseInterferenceGrid_IQ'].transpose(2, 0, 1)  # (H,W,C) to (C,H,W)
        NI_IQ = NI_IQ[:2, :, :].astype(np.float32)  # First two channels

        if NI_IQ.shape != (2, 300, 14):
            print(f"Skipping {test}: unexpected shape {NI_IQ.shape}")
            continue
        if np.isnan(NI_IQ).any() or np.isinf(NI_IQ).any():
            print(f"Warning: NaN/Inf in {test}")
            continue
        # NI_IQ = (NI_IQ - np.min(NI_IQ)) / (np.max(NI_IQ) - np.min(NI_IQ))
        # NI_IQ = NI_IQ / np.max(np.abs(NI_IQ))
        # NI_IQ = NI_IQ / IQ_NORMALIZATION_FACTOR
        # NI_IQ = np.clip(NI_IQ, -1, 1)
        tests.append(NI_IQ)

        label = 0 if ground_truth_file_path is np.nan else 1
        ground_truths.append([label])

    # Convert to tensors
    tests = torch.tensor(np.stack(tests))  # Shape: (N, 2, 300, 14)
    ground_truths = torch.tensor(np.stack(ground_truths))  # Shape: (N, 1) 
    return tests, ground_truths

def load_test_dataset_normal(test_df):
    tests = []

    for index, row in test_df.iterrows():
        test_file_path = row["file_path"]
        ground_truth_file_path = row["ground_truth_file_path"]
        if ground_truth_file_path is not np.nan:
            continue
        test = scipy.io.loadmat(test_file_path)
        NI_IQ = test['noiseInterferenceGrid_IQ'].transpose(2, 0, 1)  # (H,W,C) to (C,H,W)
        NI_IQ = NI_IQ[:2, :, :].astype(np.float32)  # First two channels
        if NI_IQ.shape != (2, 300, 14):
            print(f"Skipping {test}: unexpected shape {NI_IQ.shape}")
            continue
        if np.isnan(NI_IQ).any() or np.isinf(NI_IQ).any():
            print(f"Warning: NaN/Inf in {test}")
            continue
        # NI_IQ = NI_IQ / IQ_NORMALIZATION_FACTOR
        tests.append(NI_IQ)
    tests = torch.tensor(np.stack(tests))  # Shape: (N, 2, 300, 14)
    return tests

def load_train_dataset(train_df):
    trains = []

    for index, row in train_df.iterrows():
        train_file_path = row["file_path"]
        train = scipy.io.loadmat(train_file_path)
        NI_IQ = train['noiseInterferenceGrid_IQ'].transpose(2, 0, 1)  # (H,W,C) to (C,H,W)
        NI_IQ = NI_IQ[:2, :, :].astype(np.float32)  # First two channels
        # NI_IQ = (NI_IQ - np.min(NI_IQ)) / (np.max(NI_IQ) - np.min(NI_IQ))
        # NI_IQ = NI_IQ / np.max(np.abs(NI_IQ))
        # NI_IQ = NI_IQ / IQ_NORMALIZATION_FACTOR
        # NI_IQ = np.clip(NI_IQ, -1, 1)
        trains.append(NI_IQ)
    
    # max_train = np.max(np.abs(trains))
    # print("max_train: ", max_train)
    # trains = trains / max_train
    # print("trains.min: ", trains.min())
    # print("trains.max: ", trains.max())
    # Convert to tensors
    trains = torch.tensor(np.stack(trains))  # Shape: (N, 2, 300, 14)
    return trains

if __name__ == "__main__":
    print("Loading train dataset")
    TRAIN_DF_FILE_PATH = os.path.join(os.path.dirname(__file__), "../datasets/FccIQ/synthetic/train_df.csv")
    print("TRAIN_DF_FILE_PATH: ", TRAIN_DF_FILE_PATH)
    train_df = pd.read_csv(TRAIN_DF_FILE_PATH)
    print(f'train_df.shape: {train_df.shape}')
    trains = load_train_dataset(train_df)
    print(f'trains.shape: {trains.shape}')
    print("trains.min: ", trains.min())
    print("trains.max: ", trains.max())

    print("Loading test dataset")
    TEST_DF_FILE_PATH = os.path.join(os.path.dirname(__file__), "../datasets/FccIQ/synthetic/test_df.csv")
    print("TEST_DF_FILE_PATH: ", TEST_DF_FILE_PATH)
    test_df = pd.read_csv(TEST_DF_FILE_PATH)
    print(f'test_df.shape: {test_df.shape}')
    tests, ground_truths = load_test_dataset(test_df)
    print(f'tests.shape: {tests.shape}')
    print(f'ground_truths.shape: {ground_truths.shape}')
    # print(f'ground_truths: {ground_truths}')
    print("tests.min: ", tests.min())
    print("tests.max: ", tests.max())
