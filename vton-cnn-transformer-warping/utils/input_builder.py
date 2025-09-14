import torch

def build_input(batch, use_cloth=False, use_cloth_mask=False):
    """
    Build model input tensor dari batch dataloader.

    Args:
        batch (dict): dict dari dataloader
        use_cloth (bool): tambahkan cloth_img ke input
        use_cloth_mask (bool): tambahkan cloth_mask ke input

    Return:
        Tensor (B,C,H,W): input ke model
    """
    inputs = [batch["person_img"], batch["pose"]]

    # pastikan parse selalu (B,1,H,W)
    parse = batch["parse"]
    if parse.ndim == 3:
        parse = parse.unsqueeze(1)
    inputs.append(parse)

    if use_cloth and "cloth_img" in batch:
        inputs.append(batch["cloth_img"])
    if use_cloth_mask and "cloth_mask" in batch:
        mask = batch["cloth_mask"]
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        inputs.append(mask)

    return torch.cat(inputs, dim=1)


def get_in_channels_from_batch(batch, use_cloth=False, use_cloth_mask=False):
    """Hitung otomatis jumlah channel input dari satu batch."""
    x = build_input(batch, use_cloth, use_cloth_mask)
    return x.shape[1]
