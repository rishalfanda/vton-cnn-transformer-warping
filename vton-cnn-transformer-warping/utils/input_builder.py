import torch

def build_input(batch, use_cloth=False, use_cloth_mask=False):
    """
    Build model input tensor dari batch dataloader.

    Args:
        batch (dict): dict dari dataloader, berisi:
            - person_img : (B,3,H,W)
            - pose       : (B,18,H,W)
            - parse      : (B,H,W) atau (B,1,H,W)
            - cloth_img  : (B,3,H,W) [optional]
            - cloth_mask : (B,1,H,W) [optional]
        use_cloth (bool): apakah tambahkan cloth_img ke input
        use_cloth_mask (bool): apakah tambahkan cloth_mask ke input

    Return:
        x (Tensor): input ke model, shape (B,C,H,W)
    """
    inputs = [batch["person_img"], batch["pose"]]

    parse = batch["parse"]
    if parse.ndim == 3:  # (B,H,W)
        parse = parse.unsqueeze(1)  # jadi (B,1,H,W)
    inputs.append(parse)

    if use_cloth and "cloth_img" in batch:
        inputs.append(batch["cloth_img"])
    if use_cloth_mask and "cloth_mask" in batch:
        mask = batch["cloth_mask"]
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        inputs.append(mask)

    x = torch.cat(inputs, dim=1)
    return x
