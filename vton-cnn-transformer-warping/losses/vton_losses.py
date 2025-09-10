
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
        targets = targets.clone().long()   # ✅ clone supaya aman
        num_classes = inputs.shape[1]

        # clamp target supaya aman
        targets[targets >= num_classes] = 0
        targets[targets < 0] = 0

        inputs = F.softmax(inputs, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0,3,1,2).float()

        intersection = (inputs * targets_onehot).sum(dim=(2,3))
        union = inputs.sum(dim=(2,3)) + targets_onehot.sum(dim=(2,3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()
        return loss


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
        targets = targets.clone().long()   # ✅ clone supaya aman
        targets[targets >= num_classes] = 0
        targets[targets < 0] = 0

        inputs = F.softmax(inputs, dim=1)
        preds = torch.argmax(inputs, dim=1, keepdim=True).float()
        targets = targets.unsqueeze(1).float()

        # 🔥 pastikan kernel ikut device & dtype
        self.laplace = self.laplace.to(device=preds.device, dtype=preds.dtype)

        pred_boundary = torch.sigmoid(self.laplace(preds))
        target_boundary = torch.sigmoid(self.laplace(targets))

        return F.l1_loss(pred_boundary, target_boundary)


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
        targets = targets.clone().long()   # ✅ clone supaya aman
        targets[targets >= self.num_classes] = 0
        targets[targets < 0] = 0

        dice = self.dice_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        loss = self.dice_w * dice + self.ce_w * ce

        if self.use_boundary_loss:
            boundary = self.boundary_loss(inputs, targets)
            loss += self.boundary_w * boundary
            return loss, {"dice": dice.item(), "ce": ce.item(), "boundary": boundary.item()}
        else:
            return loss, {"dice": dice.item(), "ce": ce.item()}

