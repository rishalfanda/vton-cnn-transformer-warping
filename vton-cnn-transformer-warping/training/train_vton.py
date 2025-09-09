import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from dataloader.vton_data_loader import create_dataloaders
from utils.input_builder import build_input
from losses.vton_losses import SegmentationLoss
from models.vton_unet import UNet
from models.vton_transunet import TransUNetLight
from configs.vton_config_default import config


def train_one_epoch(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        inputs = build_input(batch).to(device)
        targets = batch["parse"].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss, loss_dict = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = build_input(batch).to(device)
            targets = batch["parse"].to(device)

            with autocast():
                outputs = model(inputs)
                loss, _ = criterion(outputs, targets)

            total_loss += loss.item()
    return total_loss / len(val_loader)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # --- Data ---
    train_loader, val_loader = create_dataloaders(config, batch_size_override=config["training"]["batch_size"])

    # --- Model (choose baseline or transunet-light) ---
    model_type = config["segmentation"]["model_type"]
    if model_type == "unet":
        model = UNet(in_channels=22, num_classes=20).to(device)
    else:
        model = TransUNetLight(in_channels=22, num_classes=20).to(device)

    # --- Loss ---
    criterion = SegmentationLoss(
        num_classes=20,
        dice_weight=config["segmentation"]["dice_weight"],
        ce_weight=config["segmentation"]["ce_weight"],
        boundary_weight=config["segmentation"]["boundary_weight"],
        use_boundary_loss=config["segmentation"]["use_boundary_loss"]
    )

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(),
                           lr=config["training"]["learning_rate"],
                           weight_decay=config["training"]["weight_decay"])

    # --- AMP Scaler ---
    scaler = GradScaler()

    # --- Training loop ---
    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{config['training']['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save checkpoint tiap beberapa epoch
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            ckpt_path = os.path.join(config["training"]["checkpoint_dir"], f"model_epoch{epoch+1}.pth")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
