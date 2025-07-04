import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
    
DATASET_DIR = os.path.join(os.path.dirname(__file__), "../datasets/FccIQ/synthetic/")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
GOOD_DIR = os.path.join(TRAIN_DIR, "good")

TEST_DIR = os.path.join(DATASET_DIR, "test")
BAD_DIR = os.path.join(TEST_DIR, "singletone")

GROUND_TRUTH_DIR = os.path.join(DATASET_DIR, "ground_truth", "singletone")

print("-"*100)
print("DATASET_DIR: ", DATASET_DIR)
print("-"*10)
print("TRAIN_DIR: ", TRAIN_DIR)
print("GOOD_DIR: ", GOOD_DIR)
print("-"*10)
print("TEST_DIR: ", TEST_DIR)
print("BAD_DIR: ", BAD_DIR)
print("GROUND_TRUTH_DIR: ", GROUND_TRUTH_DIR)
print("-"*100)

def build_df():
    good_files = glob.glob(os.path.join(GOOD_DIR, "*NoiseInterferenceGrid.mat"))
    bad_files = glob.glob(os.path.join(BAD_DIR, "*NoiseInterferenceGrid.mat"))
    ground_truth_files = glob.glob(os.path.join(GROUND_TRUTH_DIR, "*InterferenceGrid.mat"))

    # print("good_files: ", good_files)
    # print("bad_files: ", bad_files)
    # print("ground_truth_files: ", ground_truth_files)

    good_list = []
    for good_file_path in good_files:
        good_file_name = good_file_path.split("/")[-1]
        config = good_file_name.split(".")[0].split("_")
        SNR = int(config[1])
        MCS = int(config[3])
        Slot = int(config[5])
        row = [SNR, -1, MCS, -1, Slot, good_file_path, None]
        good_list.append(row)

    bad_list = []
    for bad_file_path in bad_files:
        bad_file_name = bad_file_path.split("/")[-1]
        config = bad_file_name.split(".")[0].split("_")
        SNR = int(config[1])
        SIR = int(config[3])
        MCS = int(config[5])
        FRQ = int(config[7])
        Slot = int(config[9])   
        ground_truth_file_path = os.path.join(GROUND_TRUTH_DIR, bad_file_name.replace("NoiseInterferenceGrid", "InterferenceGrid"))
        if ground_truth_file_path not in ground_truth_files:
            print("ground_truth_file_path not found: ", ground_truth_file_path)
            continue
        row = [SNR, SIR, MCS, FRQ, Slot, bad_file_path, ground_truth_file_path]
        bad_list.append(row)

    good_df = pd.DataFrame(good_list, columns=["SNR", "SIR", "MCS", "FRQ", "Slot", "file_path", "ground_truth_file_path"])
    bad_df = pd.DataFrame(bad_list, columns=["SNR", "SIR", "MCS", "FRQ", "Slot", "file_path", "ground_truth_file_path"])
    
    # Save DataFrames to CSV files
    good_df.to_csv(os.path.join(DATASET_DIR, "good_df.csv"), index=False)
    bad_df.to_csv(os.path.join(DATASET_DIR, "bad_df.csv"), index=False)

    # Split good data into train and test (80-20 split)
    good_train_df, good_test_df = train_test_split(good_df, test_size=0.2, random_state=42)

    # Save train and test splits
    good_train_df.to_csv(os.path.join(DATASET_DIR, "good_train_df.csv"), index=False)
    good_test_df.to_csv(os.path.join(DATASET_DIR, "good_test_df.csv"), index=False)
    
    # For bad data, we'll use all of it as test data since it represents anomalies
    bad_test_df = bad_df.copy()
    bad_test_df.to_csv(os.path.join(DATASET_DIR, "bad_test_df.csv"), index=False)
    
    print(f"Good train samples: {good_train_df.shape}")
    print(f"Good test samples: {good_test_df.shape}")
    print(f"Bad test samples: {bad_test_df.shape}")

    train_df = good_train_df
    test_df = pd.concat([good_test_df, bad_test_df])

    train_df.sort_values(by=["SNR", "MCS", "Slot"], inplace=True)
    test_df.sort_values(by=["SNR", "MCS", "Slot"], inplace=True)

    train_df_file_path = os.path.join(DATASET_DIR, "train_df.csv")
    test_df_file_path = os.path.join(DATASET_DIR, "test_df.csv")
    print("Train df file path: ", train_df_file_path)
    print("Test df file path: ", test_df_file_path)

    # Save train and test splits
    train_df.to_csv(train_df_file_path, index=False)
    test_df.to_csv(test_df_file_path, index=False)
    
    print(f"Train dataset: {train_df.shape}")
    print(f"Test dataset: {test_df.shape} ({good_test_df.shape[0]} + {bad_test_df.shape[0]} = {good_test_df.shape[0] + bad_test_df.shape[0]})")

    return train_df, test_df

if __name__ == "__main__":
    build_df()
