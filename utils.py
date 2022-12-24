import torch
import torch.nn.functional as F


def binary_dice_metric(logit, target, threshold=0.5, smooth = 1e-8, norm = True):

    if target.ndim == 3:
        target = target.unsqueeze(0)

    assert logit.shape == target.shape
    
    logit = logit.sigmoid()
    if norm:
        logit = (logit - logit.min()) / (logit.max() - logit.min() + smooth)

    pred = torch.zeros_like(target)
    pred[logit >= threshold] = 1

    intersection = (pred == 1) & (target == 1)
    intersection = torch.sum(intersection == 1)
    denominator = torch.sum(pred) + torch.sum(target)
    dice = 2. * intersection / denominator
    return dice


# def binary_dice_metric(logits, target, threshold=0.5, smooth = 1e-8):

#     assert logits.shape == target.shape
#     assert target.ndim == 4, f"make sure the tensor shape is [N,C,H,W]."

#     logits = torch.where(torch.sigmoid(logits) > threshold, 1, 0) # 
#     logits = logits.contiguous()
#     target = target.contiguous()    

#     intersection = torch.sum(logits * target + smooth, dim=[1,2,3])
#     denominator = torch.sum(logits + smooth, dim=[1,2,3]) + torch.sum(target + smooth, dim=[1,2,3])
    
#     return torch.mean(2. * intersection / denominator)

def threshold_binary_dice_metrics(logit, target, norm=True, up_mode=False, smooth = 1e-8):

    # if torch.is_floating_point(target):
    #     target = target.type(torch.uint8)

    if target.ndim == 3: 
        target = target.unsqueeze(0)

    if up_mode:
        logit = F.interpolate(logit, size=target.shape[-2:], \
            mode='bilinear', align_corners=False)
    
    assert logit.shape == target.shape, \
        f"make sure the logit shape {logit.shape} match with target shape: {target.shape}."
    
    logit = logit.sigmoid()
    if norm:
        logit = (logit - logit.min()) / (logit.max() - logit.min() + smooth)

    Thresholds = torch.linspace(1, 0, 256, device=target.device)[:, None, None, None]
    logits_nd = logit.repeat(256,1,1,1)
    target_nd = target.repeat(256,1,1,1)
    pred_nd = torch.zeros_like(target_nd)
    pred_nd[logits_nd >= Thresholds] = 1

    intersection = (pred_nd == 1) & (target_nd == 1)
    intersection_thres =  torch.sum(intersection == 1, dim=[1,2,3])
    denominator_thres = torch.sum(pred_nd, dim=[1,2,3]) + torch.sum(target_nd, dim=[1,2,3])
    dice_thres =  2. * intersection_thres / denominator_thres

    return torch.mean(dice_thres)


# def threshold_binary_dice_metrics_v0(logit, target, norm=True, up_mode=False):

#     if not set(torch.unique(target).tolist()).issubset({0,1}):
#         target = (target > 0.5).type(torch.uint8)

#     if torch.is_floating_point(target):
#         target = target.type(torch.uint8)

#     if target.ndim == 3: 
#         target = target.unsqueeze(0)

#     if up_mode:
#         logit = F.interpolate(logit, size=target.shape[-2:], \
#             mode='bilinear', align_corners=False)
    
#     assert logit.shape == target.shape, \
#         f"make sure the logit shape {logit.shape} match with target shape: {target.shape}."
    
#     logit = logit.sigmoid()
#     if norm:
#         logit = (logit - logit.min()) / (logit.max() - logit.min() + 1e-8)

#     Thresholds = torch.linspace(1, 0, 256, device=target.device)[:, None, None, None]
#     logits_nd = logit.repeat(256,1,1,1)
#     target_nd = target.repeat(256,1,1,1)
#     pred_nd = torch.zeros_like(target_nd)
#     pred_nd[logits_nd >= Thresholds] = 1

#     intersection = pred_nd & target_nd
#     intersection_thres =  torch.sum(intersection == 1, dim=[1,2,3])
#     denominator_thres = torch.sum(pred_nd, dim=[1,2,3]) + torch.sum(target_nd, dim=[1,2,3])
#     dice_thres =  2. * intersection_thres / denominator_thres

#     return torch.mean(dice_thres)