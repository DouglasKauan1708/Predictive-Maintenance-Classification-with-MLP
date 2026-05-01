import kagglehub
import shutil, os
import wandb
import pandas as pd

# Downloading the dataset
def data_downloading():
    path = kagglehub.dataset_download("shivamb/machine-predictive-maintenance-classification")
    destination = os.getcwd()
    raw_path = os.path.join(destination, "data", "raw")
    shutil.copytree(path,raw_path, dirs_exist_ok = True)

    return raw_path

# Loading the dataset
def load_data(path="data/raw/predictive_maintenance.csv"):
    df = pd.read_csv(path)
    return df