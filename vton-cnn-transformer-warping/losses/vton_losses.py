import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Dice Loss
# ---------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (B,C,H,W), logits
        targets = targets.clone().long()
        num_classes = inputs.shape[1]

        # amankan nilai target
        targets = torch.clamp(targets, 0, num_classes - 1)

        # softmax -> one-hot
        probs = F.softmax(inputs, dim=1).clamp(min=1e-7, max=1.0)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


# ---------------------------
# Boundary-Aware Loss
# ---------------------------
class BoundaryAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        laplace_kernel = torch.tensor([[[[0,-1,0],
                                         [-1,4,-1],
                                         [0,-1,0]]]], dtype=torch.float32)
        self.laplace = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.laplace.weight = nn.Parameter(laplace_kernel, requires_grad=False)

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        targets = targets.clone().long()
        targets[targets >= num_classes] = 0
        targets[targets < 0] = 0

        inputs = F.softmax(inputs, dim=1)
        preds = torch.argmax(inputs, dim=1, keepdim=True).float()
        targets = targets.unsqueeze(1).float()

        # pastikan kernel ikut device & dtype
        self.laplace = self.laplace.to(device=preds.device, dtype=preds.dtype)

        pred_boundary = self.laplace(preds)
        target_boundary = self.laplace(targets)

        # gunakan BCE with logits (lebih aman untuk autocast AMP)
        return F.binary_cross_entropy_with_logits(pred_boundary, target_boundary)


# ---------------------------
# Combined Segmentation Loss
# ---------------------------
class SegmentationLoss(nn.Module):
    def __init__(self, num_classes=20,
                 dice_weight=0.4, ce_weight=0.3, boundary_weight=0.3,
                 use_boundary_loss=True):
        super().__init__()
        self.num_classes = num_classes
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.boundary_loss = BoundaryAwareLoss() if use_boundary_loss else None
        self.dice_w = dice_weight
        self.ce_w = ce_weight
        self.boundary_w = boundary_weight
        self.use_boundary_loss = use_boundary_loss

    def forward(self, inputs, targets):
        targets = targets.clone().long()
        targets = torch.clamp(targets, 0, self.num_classes - 1)

        dice = self.dice_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        total_loss = self.dice_w * dice + self.ce_w * ce

        loss_dict = {"dice": dice.item(), "ce": ce.item()}

        if self.use_boundary_loss:
            boundary = self.boundary_loss(inputs, targets)
            total_loss += self.boundary_w * boundary
            loss_dict["boundary"] = boundary.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


# ---------------------------
# Quick test
# ---------------------------
if __name__ == "__main__":
    model_loss = SegmentationLoss(num_classes=20, use_boundary_loss=True)
    x = torch.randn(2, 20, 512, 384)  # logits
    y = torch.randint(0, 20, (2, 512, 384))  # target
    loss, logs = model_loss(x, y)
    print("Loss:", loss.item())
    print("Logs:", logs)
