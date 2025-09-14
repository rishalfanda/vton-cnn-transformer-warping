import torch
from dataloader.vton_data_loader import create_dataloaders
from configs.vton_config_default import config

def check_batch(loader, name="train"):
    batch = next(iter(loader))
    print(f"\n{name.upper()} batch keys:", batch.keys())
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} ({v.dtype})")
        else:
            print(f"  {k}: {type(v)}")

if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders(config, batch_size_override=2)

    print("=== TRAIN ===")
    check_batch(train_loader, "train")

    print("=== VAL ===")
    check_batch(val_loader, "val")
