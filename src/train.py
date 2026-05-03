from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from src.model import MLP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.losses import FocalLoss

def prepare_dataloaders (train_df, test_df, target_col: str, batch_size: int):

    # Separate features and labels
    X_train = train_df.drop(columns = [target_col]).values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.int64)
    

    X_test = test_df.drop(columns = [target_col]).values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.int64)

    # Standardize : fit on train , transform both
    scaler = StandardScaler ()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # NO fit here !

    class_counts = np.bincount(y_train)
    class_weights = 1.0 / np.sqrt(class_counts)

    sample_weights = class_weights[y_train]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=int(len(sample_weights) * 0.6),
    replacement=True
    )

    # Build TensorDatasets
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train ))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test ))

    # Wrap in DataLoaders
    train_loader = DataLoader(train_ds, batch_size = batch_size, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader, scaler

def train_model(config : dict, train_loader, test_loader, input_dim: int) -> nn.Module:

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim = input_dim, hidden_sizes = config ["model"]["hidden_sizes"], dropout = config ["model"]["dropout"]).to(device)
    y_train = train_loader.dataset.tensors[1].cpu().numpy()
    class_counts = np.bincount(y_train)
    criterion = FocalLoss(gamma=1.5)
    optimizer = optim.Adam(model.parameters(), lr = config["model"]["learning_rate"])

    # Tell W&B to track gradients and parameter histograms
    wandb.watch(model, log = "all", log_freq = 10)

    best_val_loss = float("inf")
    patience = config["model"]["early_stopping_patience"]
    patience_counter = 0

    for epoch in range(config ["model"]["epochs"]):

        # Training phase
        model.train()        # activates Dropout
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()        # clear previous gradients
            output = model(X_batch)      # forward pass
            loss = criterion(output, y_batch)
            loss.backward()              # compute gradients
            optimizer.step()             # update weights

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()        # deactivates Dropout
        val_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():       # no gradient computation
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * X_batch.size(0)

                pred = torch.argmax(output, dim=1)
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        val_loss /= len(test_loader.dataset)
        val_acc = accuracy_score(y_true, y_pred)
        val_prec = precision_score(y_true, y_pred, average="macro")
        val_rec = recall_score(y_true, y_pred, average="macro")
        val_f1 = f1_score(y_true, y_pred, average="macro")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d")
        
        # W&B logging
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_prec": val_prec,
            "val_rec": val_rec,
            "val_f1": val_f1,
            "confusion_matrix": wandb.Image(plt),
        })

        # Early stopping
        if val_loss < best_val_loss :
            best_val_loss = val_loss
            patience_counter = 0

            # Save the best model checkpoint
            torch.save(model.state_dict(), "best_model.pt")

            # Log checkpoint as a W&B artifact
            model_artifact = wandb.Artifact ("trained_model", type = "model", description = f" MLP checkpoint at epoch { epoch }")
            model_artifact.add_file("best_model.pt")
            wandb.log_artifact(model_artifact)

        else :
            patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}." f"Best val_loss: {best_val_loss:.4f}")
                break

    # Load the best weights before returning
    model.load_state_dict(torch.load("best_model.pt"))
    return model