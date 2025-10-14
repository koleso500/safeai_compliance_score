import json
import itertools
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from old_code.torch_for_credits.torch_model import NeuralNetwork

# Data separation
data_lending_ny_clean = pd.read_csv("../saved_data/data_lending_clean_ny_article.csv")
x = data_lending_ny_clean.drop(columns=['response'])
y = data_lending_ny_clean['response']

# Splitting into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)

#GPU connection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# Ensure cuDNN is optimized
torch.backends.cudnn.benchmark = True

# Standardize data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test) #important not to fit again(mean and standard deviation)
x_train_scaled_names = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)
x_test_scaled_names = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)

# Convert to PyTorch tensors
torch.manual_seed(42)
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1) #1D (n) -> 2D (n,1)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Hyperparameters tuning
batch_sizes = [64, 128, 256]
max_lr = 0.001
hidden_layer_sizes = [64, (32,16), (64, 32, 16), (128, 64, 32)]
dropout_rates = [0.2, 0.3]

# Preparation
k_folds = 2
kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
best_f1 = 0
best_threshold = 0
best_params = None
best_model_path = None
best_train_losses = []
best_val_losses = []
max_fold_f1 = 0
model = None
model_dir = "../saved_models"

# Automatic mixed precision
scaler_amp = torch.amp.GradScaler(device="cuda")

# Learning
for batch_size, layers, dropout_rate in itertools.product(batch_sizes, hidden_layer_sizes,
                                                              dropout_rates):
    fold_accuracies = []
    fold_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_tensor, y_train_tensor)):

        # Slicing
        train_idx_tensor = torch.tensor(train_idx, dtype=torch.long, device="cpu")
        val_idx_tensor = torch.tensor(val_idx, dtype=torch.long, device="cpu")
        x_train_fold_tensor = x_train_tensor[train_idx_tensor].to(device)
        y_train_fold_tensor = y_train_tensor[train_idx_tensor].to(device)
        x_val_fold_tensor = x_train_tensor[val_idx_tensor].to(device)
        y_val_fold_tensor = y_train_tensor[val_idx_tensor].to(device)

        # DataLoader
        train_dataset = torch.utils.data.TensorDataset(x_train_fold_tensor, y_train_fold_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # Define model
        model = NeuralNetwork(x_train.shape[1], layers, dropout_rate).to(device)
        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.Adam(model.parameters())

        # 1Cycle Learning Rate
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=100,
                               pct_start=0.3, anneal_strategy='cos')

        # Early Stopping Setup
        patience = 10
        best_val_loss = float('inf')
        counter = 0
        fold_train_losses = []
        fold_val_losses = []

        for epoch in range(100):  # Train for up to 100 epochs
            model.train()
            epoch_loss = 0
            first_batch = True
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)  # Move batch to GPU
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):  # Use AMP for mixed precision
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
                if not first_batch:  # Scheduler doesn't step in the first batch
                    scheduler.step()
                first_batch = False  # Scheduler updates for the next batches

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val_fold_tensor)
                val_loss = criterion(val_outputs, y_val_fold_tensor).item()

            fold_train_losses.append(avg_train_loss)
            fold_val_losses.append(val_loss)

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Evaluate on validation set
        with torch.no_grad():
            y_val_pred = model(x_val_fold_tensor)

            # Apply sigmoid to convert to probabilities if needed
            if y_val_pred.shape[1] == 1:
                y_val_pred_prob = torch.sigmoid(y_val_pred).cpu().numpy()
            else:
                y_val_pred_prob = y_val_pred.cpu().numpy()

            # Convert to numpy for evaluation
            y_val_fold_numpy = y_val_fold_tensor.cpu().numpy()

            # Flatten
            y_val_pred_prob_flat = y_val_pred_prob.flatten()

            for threshold in np.arange(0.0, 1.01, 0.01):
                y_val_pred_labels = np.where(y_val_pred_prob_flat > threshold, 1, 0)
                f1 = f1_score(y_val_fold_numpy, y_val_pred_labels)
                fold_f1_scores.append(f1)

                # Update best F1 score and corresponding threshold
                if f1 > max_fold_f1:
                    max_fold_f1 = f1
                    best_threshold = threshold

            # Track best model
            if max_fold_f1 > best_f1:
                best_f1 = max_fold_f1
                best_params = (batch_size, max_lr, layers, dropout_rate)
                best_model_probabilities = y_val_pred_prob
                best_train_losses = fold_train_losses.copy()
                best_val_losses = fold_val_losses.copy()

                # Save the best model
                best_torch_model_path = os.path.join(model_dir, "best_torch_model_ny_article.pth")
                if model is not None:
                    torch.save(model.state_dict(), best_torch_model_path)
                    print(f"New best model saved at {model_dir}")
                else:
                    print("Model was not initialized")

print("\nBest Hyperparameters:", best_params, "with F1:", best_f1)
print(f"Best model saved at {model_dir}")

# Save variables to the folder
# Tensors
tensor_path = os.path.join("../saved_data", "full_data_tensors_ny_article.pth")
torch.save({
    "x_train_tensor_ny_article": x_train_tensor,
    "y_train_tensor_ny_article": y_train_tensor,
    "x_test_tensor_ny_article": x_test_tensor,
    "y_test_tensor_ny_article": y_test_tensor,
}, tensor_path)

# Best parameters
json_str = json.dumps(best_params, indent=4)
file_path = os.path.join("../saved_data", "best_torch_params_ny_article.json")
with open(file_path, "w", encoding="utf-8") as file:
    file.write(json_str)
print("Best parameters saved successfully!")

# Other
x_train_scaled_names.to_csv(os.path.join("../saved_data", "x_train_scaled_names_ny_article.csv"), index=False)
x_test_scaled_names.to_csv(os.path.join("../saved_data", "x_test_scaled_names_ny_article.csv"), index=False)
np.save(os.path.join("../saved_data", "y_train_ny_article.npy"), y_train)
np.save(os.path.join("../saved_data", "y_test_ny_article.npy"), y_test)
np.save("../saved_data/best_train_losses_ny_article.npy", np.array(best_train_losses))
np.save("../saved_data/best_val_losses_ny_article.npy", np.array(best_val_losses))