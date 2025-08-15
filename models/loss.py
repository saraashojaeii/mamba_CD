import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor, einsum
from misc.torchutils import class2one_hot,simplex
from models.darnet_help.loss_help import FocalLoss, dernet_dice_loss

def cross_entropy_loss_fn(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    
    # Print shapes for debugging
    # print(f"Initial shapes - Input: {input.shape}, Target: {target.shape}")
    
    # Handle channels-last format (NHWC -> NCHW)
    if target.dim() == 4 and target.shape[-1] == 3:
        # This is likely an RGB image in channels-last format
        # Take only the first channel and permute to channels-first
        target = target[..., 0]  # Take first channel -> [N, H, W]
        # print(f"Converted target from NHWC to NHW: {target.shape}")
    # Handle standard channels-first format
    elif target.dim() == 4:
        target = torch.squeeze(target, dim=1)  # N*H*W
    
    # For flattened targets, reshape them back to spatial dimensions
    if target.dim() == 2 and target.shape[1] > 1000:  # Likely a flattened spatial target
        # Calculate spatial dimensions - assuming square images for simplicity
        h = w = int(math.sqrt(target.shape[1]))
        target = target.reshape(target.shape[0], h, w)
        # print(f"Reshaped target from flattened to spatial: {target.shape}")
    
    # Get number of classes from input tensor
    num_classes = input.shape[1]
    
    # Clamp target values to valid class range (0 to num_classes-1)
    target = torch.clamp(target, 0, num_classes-1)
    # print(f"Clamped target values to range [0, {num_classes-1}]")
    
    # Now ensure spatial dimensions match for interpolation
    if input.shape[-1] != target.shape[-1] or input.shape[-2] != target.shape[-2]:
        try:
            print(f"Interpolating input from {input.shape} to match target {target.shape}")
            input = F.interpolate(input, size=(target.shape[-2], target.shape[-1]), mode='bilinear', align_corners=True)
        except Exception as e:
            print(f"Error during interpolation: {e}")
            print(f"Input shape: {input.shape}, Target shape: {target.shape}")
    
    # print(f"Final shapes - Input: {input.shape}, Target: {target.shape}")

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

class DiceLoss(nn.Module):
    def __init__(self, num_classes, weight=None, ignore_index=255, smooth=1e-10, idc=None):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.ignore_index = ignore_index 
        self.smooth = smooth
        self.idc = idc if idc is not None else list(range(self.num_classes))

    def forward(self, predicts, target):
        probs = torch.softmax(predicts, dim=1)
        
        target = target.long()
        if target.dim() == 4:
            target = target.squeeze(1)
        
        # Handle channels-last format if needed
        if target.dim() == 3 and target.shape[-1] == 3:
            target = target[..., 0]  # Take first channel
            # print(f"DiceLoss: Converted target from channels-last format: {target.shape}")
        
        # Clamp target values to valid class range (0 to num_classes-1)
        target = torch.clamp(target, 0, self.num_classes-1)
        # print(f"DiceLoss: Clamped target values to range [0, {self.num_classes-1}]")
        
        # Create one-hot encoding
        try:
            target_one_hot = class2one_hot(target, self.num_classes)
        except Exception as e:
            # print(f"Error in class2one_hot: {e}")
            # print(f"Target shape: {target.shape}, min: {target.min()}, max: {target.max()}")
            # Fallback: create one-hot encoding manually
            b, h, w = target.shape
            target_one_hot = torch.zeros((b, self.num_classes, h, w), device=target.device, dtype=torch.int32)
            for c in range(self.num_classes):
                target_one_hot[:, c, :, :] = (target == c)
        
        assert simplex(probs)
        assert simplex(target_one_hot)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target_one_hot[:, self.idc, ...].type(torch.float32)

        intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
        
        card_pc: Tensor = einsum("bcwh->bc", pc)
        card_tc: Tensor = einsum("bcwh->bc", tc)
        
        union: Tensor = card_pc + card_tc

        dice_score_per_class: Tensor = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss_per_class: Tensor = 1.0 - dice_score_per_class

        if self.weight is not None:
            class_weights = torch.tensor(self.weight, device=predicts.device, dtype=torch.float32)[self.idc]
            dice_loss_per_class = dice_loss_per_class * class_weights.view(1, -1) # ensure broadcasting b*c
            loss = (dice_loss_per_class.sum(dim=1) / class_weights.sum()).mean() # weighted mean over classes, then mean over batch
        else:
            loss = dice_loss_per_class.mean() # Mean over classes and batch

        return loss

class CEDiceLoss(nn.Module):
    def __init__(self, num_classes, ce_weight=0.5, dice_weight=0.5, cross_entropy_kwargs=None, dice_loss_kwargs=None):
        super(CEDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        _ce_kwargs = cross_entropy_kwargs if cross_entropy_kwargs is not None else {}
        self.cross_entropy_fn = cross_entropy_loss_fn

        _dice_kwargs = dice_loss_kwargs if dice_loss_kwargs is not None else {}
        self.dice_loss = DiceLoss(num_classes=self.num_classes, **_dice_kwargs)

    def forward(self, input, target):
        ce_loss = self.cross_entropy_fn(input, target)
        dice_val = self.dice_loss(input, target)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_val
        return loss

class DiceOnlyLoss(nn.Module):
    def __init__(self, num_classes, dice_loss_kwargs=None):
        super(DiceOnlyLoss, self).__init__()
        self.num_classes = num_classes
        _dice_kwargs = dice_loss_kwargs if dice_loss_kwargs is not None else {}
        self.dice_loss = DiceLoss(num_classes=self.num_classes, **_dice_kwargs)

    def forward(self, input, target):
        return self.dice_loss(input, target)

class CE2Dice1Loss(nn.Module):
    def __init__(self, num_classes, ce_weight=1.0, dice_weight=0.5, cross_entropy_kwargs=None, dice_loss_kwargs=None):
        super(CE2Dice1Loss, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        _ce_kwargs = cross_entropy_kwargs if cross_entropy_kwargs is not None else {}
        self.cross_entropy_fn = cross_entropy_loss_fn

        _dice_kwargs = dice_loss_kwargs if dice_loss_kwargs is not None else {}
        self.dice_loss = DiceLoss(num_classes=self.num_classes, **_dice_kwargs)

    def forward(self, input, target):
        ce_loss = self.cross_entropy_fn(input, target)
        dice_val = self.dice_loss(input, target)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_val
        return loss

class CE1Dice2Loss(nn.Module):
    def __init__(self, num_classes, ce_weight=0.5, dice_weight=1.0, cross_entropy_kwargs=None, dice_loss_kwargs=None):
        super(CE1Dice2Loss, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        _ce_kwargs = cross_entropy_kwargs if cross_entropy_kwargs is not None else {}
        self.cross_entropy_fn = cross_entropy_loss_fn

        _dice_kwargs = dice_loss_kwargs if dice_loss_kwargs is not None else {}
        self.dice_loss = DiceLoss(num_classes=self.num_classes, **_dice_kwargs)
        
    def forward(self, input, target):
        ce_loss = self.cross_entropy_fn(input, target)
        dice_val = self.dice_loss(input, target)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_val
        return loss

# Note: ce_scl was identical to ce_dice. If it needs specific SCL logic, it requires a separate implementation.
# For now, if 'ce_scl' is chosen, it would need to be mapped to CEDiceLoss or a new SCL specific class.


class MultiClassCDLoss(nn.Module):
    """
    Multi-class Change Detection Loss.
    Computes segmentation loss for T1, T2, and transition/change head.
    - seg_loss: loss function for segmentation ("ce", "dice", "cedice")
    - change_loss: loss function for transition/change ("ce", "cedice")
    - loss_weights: dict with weights for each head ("seg_t1", "seg_t2", "change")
    Usage:
        loss_fn = MultiClassCDLoss(num_classes, seg_loss="cedice", change_loss="ce")
        loss = loss_fn(preds, targets)
        # preds: (seg_logits_t1, seg_logits_t2, change_logits)
        # targets: dict with keys: "seg_t1", "seg_t2", "change"
    """
    def __init__(self, num_classes, seg_loss="cedice", change_loss="ce", loss_weights=None):
        super().__init__()
        self.num_classes = num_classes
        if seg_loss == "ce":
            self.seg_loss_fn = cross_entropy_loss_fn
        elif seg_loss == "dice":
            self.seg_loss_fn = DiceLoss(num_classes)
        else:
            self.seg_loss_fn = CEDiceLoss(num_classes)

        if change_loss == "ce":
            self.change_loss_fn = cross_entropy_loss_fn
        else:
            self.change_loss_fn = CEDiceLoss(num_classes*num_classes)

        self.loss_weights = loss_weights if loss_weights is not None else {"seg_t1": 1.0, "seg_t2": 1.0, "change": 1.0}

    def forward(self, preds, targets):
        # Unpack predictions
        if isinstance(preds, tuple) and len(preds) == 3:
            seg_logits_t1, seg_logits_t2, change_logits = preds
        else:
            raise ValueError(f"Expected preds to be a tuple of 3 tensors, got {type(preds)}")
        
        # Unpack targets and ensure they're properly formatted
        try:
            seg_t1 = targets["seg_t1"]
            seg_t2 = targets["seg_t2"]
            change = targets["change"]
            
            # Print debug info
            # print(f"Target shapes - seg_t1: {seg_t1.shape}, seg_t2: {seg_t2.shape}, change: {change.shape}")
            # print(f"Prediction shapes - seg_t1: {seg_logits_t1.shape}, seg_t2: {seg_logits_t2.shape}, change: {change_logits.shape}")
            
            # Get number of classes
            num_classes = seg_logits_t1.shape[1]
            # print(f"Number of classes: {num_classes}")
            
            # Process targets for segmentation (T1)
            if seg_t1.dim() == 4 and seg_t1.shape[-1] == 3:  # Handle channels-last format
                seg_t1 = seg_t1[..., 0]  # Take first channel
                # print(f"Converted seg_t1 from channels-last format: {seg_t1.shape}")
            
            # Process targets for segmentation (T2)
            if seg_t2.dim() == 4 and seg_t2.shape[-1] == 3:  # Handle channels-last format
                seg_t2 = seg_t2[..., 0]  # Take first channel
                # print(f"Converted seg_t2 from channels-last format: {seg_t2.shape}")
            
            # Process targets for change detection
            if change.dim() == 4 and change.shape[-1] == 3:  # Handle channels-last format
                change = change[..., 0]  # Take first channel
                # print(f"Converted change from channels-last format: {change.shape}")
            
            # Compute losses
            loss_t1 = self.seg_loss_fn(seg_logits_t1, seg_t1)
            loss_t2 = self.seg_loss_fn(seg_logits_t2, seg_t2)
            loss_change = self.change_loss_fn(change_logits, change)
            
        except Exception as e:
            # print(f"Error processing targets: {e}")
            # print(f"Target keys: {targets.keys()}")
            # for k, v in targets.items():
            #     if isinstance(v, torch.Tensor):
            #         print(f"  {k}: shape={v.shape}, dtype={v.dtype}, min={v.min()}, max={v.max()}")
            #     else:
            #         print(f"  {k}: {type(v)}")
            raise
        
        # Combine losses with weights
        total = (
            self.loss_weights["seg_t1"] * loss_t1 +
            self.loss_weights["seg_t2"] * loss_t2 +
            self.loss_weights["change"] * loss_change
        )
        
        return total, {"seg_t1": loss_t1.item(), "seg_t2": loss_t2.item(), "change": loss_change.item()}


def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.25, weight_neg=0.75):
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert (logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit.float(), truth.float(), reduction='none')

    pos = (truth > 0.5).float()
    neg = (truth < 0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos * pos * loss / pos_num + weight_neg * neg * loss / neg_num).sum()

    return loss

class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """

    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)

    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b * h * w, c])
        x2 = torch.reshape(x2, [b * h * w, c])

        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target, [b * h * w])

        loss = self.loss_f(x1, x2, target)
        return loss

def hybrid_loss(predictions, target, weight=[0,2,0.2,0.2,0.2,0.2]):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    # focal = FocalLoss(gamma=0, alpha=None)
    # ssim = SSIM()

    for i,prediction in enumerate(predictions):

        bce = cross_entropy(prediction, target)
        dice = dice_loss(prediction, target)
        # ssimloss = ssim(prediction, target)
        loss += weight[i]*(bce + dice) #- ssimloss

    return loss

class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """
    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label[label == 1] = -1
        label[label == 0] = 1

        mask = (label != 255).float()
        distance = distance * mask

        pos_num = torch.sum((label==1).float())+0.0001
        neg_num = torch.sum((label==-1).float())+0.0001

        loss_1 = torch.sum((1+label) / 2 * torch.pow(distance, 2)) /pos_num
        loss_2 = torch.sum((1-label) / 2 *
            torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        ) / neg_num
        loss = loss_1 + loss_2
        return loss