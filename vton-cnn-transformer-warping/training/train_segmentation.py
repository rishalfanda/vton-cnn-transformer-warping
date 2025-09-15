# train_segmentation.py

import os
import csv
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from dataloader.vton_data_loader import create_dataloaders
from utils.input_builder import build_input
from losses.vton_losses import SegmentationLoss
from models.vton_unet import UNet
from models.vton_transunet import TransUNetLight
from configs.vton_config_default import config


def train_one_epoch(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    use_cloth= config["segmentation"].get("use_cloth", False)
    use_cloth_mask = config["segmentation"].get("use_cloth_mask", False)
    for batch in pbar:
        inputs = build_input(batch, use_cloth=use_cloth, use_cloth_mask=use_cloth_mask).to(device)
        targets = batch["parse"].to(device)

        optimizer.zero_grad()

        with autocast("cuda"):
            outputs = model(inputs)
            loss, loss_dict = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{loss_dict['dice']:.3f}",
            "ce": f"{loss_dict['ce']:.3f}",
            "boundary": f"{loss_dict.get('boundary', 0):.3f}"
        })

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    use_cloth= config["segmentation"].get("use_cloth", False)
    use_cloth_mask = config["segmentation"].get("use_cloth_mask", False)
    with torch.no_grad():
        for batch in val_loader:
            inputs = build_input(batch, use_cloth=use_cloth, use_cloth_mask=use_cloth_mask).to(device)
            targets = batch["parse"].to(device)

            with autocast("cuda"):
                outputs = model(inputs)
                loss, _ = criterion(outputs, targets)

            total_loss += loss.item()
    return total_loss / len(val_loader)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # --- Data ---
    train_loader, val_loader = create_dataloaders(config, batch_size_override=config["training"]["batch_size"])

    # --- Model ---
    model_type = config["segmentation"]["model_type"]
    in_ch = config["segmentation"]["in_channels"]
    num_classes = config["segmentation"]["num_classes"]

    if model_type == "unet":
        model = UNet(in_channels=in_ch, num_classes=num_classes).to(device)
    else:
        model = TransUNetLight(in_channels=in_ch, num_classes=num_classes).to(device)

    # --- Loss ---
    criterion = SegmentationLoss(
        num_classes=config["segmentation"]["num_classes"],
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
    scaler = GradScaler("cuda")

    # --- CSV logging ---
    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
    log_path = os.path.join(config["training"]["checkpoint_dir"], "training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

        # --- Training loop ---
        for epoch in range(config["training"]["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_loss = validate(model, val_loader, criterion, device)

            writer.writerow([epoch+1, train_loss, val_loss])
            f.flush()

            print(f"Epoch {epoch+1}/{config['training']['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % config["training"]["save_interval"] == 0:
                ckpt_path = os.path.join(config["training"]["checkpoint_dir"], f"model_epoch{epoch+1}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
