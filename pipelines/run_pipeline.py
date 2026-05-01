import wandb
import numpy as np
import torch
import yaml , random
import os
from src.data_loading import load_data
from src.data_cleaning import data_cleaning
from src.data_testing import test_no_missing_values, test_target_values, compare_distributions
from src.split_data import split_train_test
from src.train import prepare_dataloaders, train_model
import pandas as pd

# Fix random seeds for full reproducibility
def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed (seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed (42)

# Importing our hyperparameters from a config.yaml file
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Initializing the Wandb 
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(
    project = "mlops-wandb-demo",
    job_type = "train",
    name = "mlp_training",
    config = config         # logs all hyperparameters automatically
)

# Loading our dataset
df_raw = load_data()

# Create an artifact and attach the raw CSV
artifact = wandb.Artifact (
name = "raw_data", # artifact name ( versioned automatically )
type = "dataset", # type tag for organizational purposes
description = "Machine Predictive Maintenance Classification raw dataset from Kaggle"
)
temp_path = "temp_raw.csv"
df_raw.to_csv(temp_path, index = False)
artifact.add_file(temp_path)
# Log the artifact to W&B
wandb.log_artifact(artifact)
# Log summary statistics visible on the W&B dashboard
wandb.summary["rows"] = len(df_raw)
wandb.summary["columns"] = list(df_raw.columns)

# Cleaning our dataset
df_clean, X, y = data_cleaning(df_raw)
clean_path = "data/clean_data.csv"
df_clean.to_csv(clean_path, index=False)

# Create an artifact and attach the clean CSV
artifact = wandb.Artifact (
name = "clean_data", # artifact name ( versioned automatically )
type = "dataset", # type tag for organizational purposes
description = "Machine Predictive Maintenance Classification after cleaning"
)
temp_path = "temp_clean.csv"
df_clean.to_csv(temp_path, index = False)
artifact.add_file(temp_path)
# Log the artifact to W&B
wandb.log_artifact(artifact)
# Log summary statistics visible on the W&B dashboard
wandb.summary["rows"] = len(df_clean)
wandb.summary["columns"] = list(df_clean.columns)

# Verifying our dataset after cleaning
test_no_missing_values(df_clean)
test_target_values(df_clean, "Failure Type", [0,1,2,3,4, 5])

selected_features = ["Air temperature [K]", "Torque [Nm]", "Tool wear [min]"]
X = X[selected_features]

# Splitting
train_df, test_df = split_train_test(
    X, y,
    test_size = config["data"]["test_size"], random_state = config["data"]["random_state"]
)

# Logging Splits and Comparison Table to W&B
for split_name, split_df in [("train_data", train_df), ("test_data", test_df)]:
    path = f"temp_{split_name}.csv"
    split_df.to_csv(path, index = False)
    art = wandb.Artifact(split_name, type = "dataset")
    art.add_file(path)
    wandb.log_artifact(art)

feature_cols = [c for c in train_df.columns if c != "Failure Type"]
comp_results = compare_distributions(train_df, test_df, feature_cols)
# Log the distribution comparison as a W&B Table
comp_df = pd.DataFrame(comp_results).T.reset_index()
comp_df.columns = ["feature", "test", "statistic", "p_value"]
comp_table = wandb.Table(dataframe = comp_df)
wandb.log({"distribution_comparison": comp_table})

wandb.summary["train_size"] = len(train_df)
wandb.summary["test_size"] = len(test_df)
print("Train and test artifacts saved.")

# Preparing our dataloaders
train_loader, test_loader, scaler = prepare_dataloaders(
    train_df, test_df,
    target_col = config["data"]["target_col"],
    batch_size = config["data"]["batch_size"]
)
input_dim = train_loader.dataset.tensors[0].shape[1]
print(f" Input dimension: {input_dim} features")

# Train
model = train_model(
    config=config, 
    train_loader=train_loader, 
    test_loader=test_loader, 
    input_dim=input_dim
    )

wandb.finish ()
print("Training complete. Best model saved to W&B.")