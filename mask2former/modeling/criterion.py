# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

import copy
def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

def calculate_iou(pred_mask,gt_mask):
    intersection = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
    union = (pred_mask + gt_mask).clamp(0, 1).sum(dim=(1, 2, 3))

    
    iou = torch.where(union > 0, intersection / union, torch.tensor(0.0, device=union.device))
    return iou
    

def src_masks_to_01(mask,class_score, mask_iou):

    class_score = class_score*mask_iou
    class_score = class_score.unsqueeze(2).unsqueeze(3)
    mask = mask*class_score
    max_indices = torch.argmax(mask, dim=0) 
    binary_mask = torch.zeros_like(mask, dtype=torch.float16)
    binary_mask.scatter_(0, max_indices.unsqueeze(1), 1)
# index(1,1,256, 256)
# self (19,1,256,256)
# self[inx[i,j,k,l]][j][k][l] = 1
    return binary_mask



def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    # loss (bs, num_query, C)
    return loss.mean(1).sum() / num_boxes


def token_sigmoid_binary_focal_loss(pred_logits, targets, alpha=0.25, gamma=2.0, text_mask=None, reduction=True):
    # binary version of focal loss
    # copied from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # pred_logits: (bs, n_anchors, max_seq_len)
    # targets: (bs, n_anchors, max_seq_len)
    # text_mask: (bs, max_seq_len)
    # assert (targets.dim() == 3)
    # assert (pred_logits.dim() == 3)  # batch x from x to

    # bs, n, _ = pred_logits.shape
    if text_mask is not None:
        # assert (text_mask.dim() == 2)
        # text_mask = (text_mask > 0).unsqueeze(1) # (bs, 1, max_seq_len)
        # text_mask = text_mask.repeat(1, pred_logits.size(1), 1)  # copy along the image channel dimension. (bs, n_anchors, max_seq_len)
        pred_logits = torch.masked_select(pred_logits, text_mask)
        targets = torch.masked_select(targets, text_mask)

        # print(pred_logits.shape)
        # print(targets.shape)

    p = torch.sigmoid(pred_logits)
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction:
        return loss.sum()
    else:
        return loss




class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, generate_with_t5, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,train_class_json,focal_alpha):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            generate_with_t5: generate label and caculate language loss
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.class_generate = generate_with_t5
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight[-1] = self.eos_coef
        # self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        import json
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        self.focal_alpha = focal_alpha
    def loss_language(self, outputs, targets, indices,  num_masks):
            targets_descriptions = []
            output_features = []
            for i, (indice, x_gt) in enumerate(zip(indices, targets)):
                pred_i, tgt_j = indice
                targets_descriptions += [self.class_texts[x_gt["labels"][x_j]] for x_j in tgt_j]
                output_features.append(outputs[i][pred_i])
            output_features = torch.cat(output_features, 0).unsqueeze(1)
            output_features_att_mask = torch.ones(output_features.size()[:-1], dtype=torch.long).to(outputs.device)
            text_decoder_loss = self.class_generate(output_features, targets_descriptions, output_features_att_mask)
            return  text_decoder_loss

    def loss_labelsVL(self, outputs, targets, indices,  num_masks):
            """Classification loss (NLL)
            targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
            """
            assert 'pred_logits' in outputs
            src_logits = outputs['pred_logits']
            num_classes = 1
            idx = self._get_src_permutation_idx(indices)
            ce_mask = torch.ones_like(src_logits, device=src_logits.device)

            target_classes_onehot = torch.zeros(src_logits.size(),
                                    dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # (bs, num_query, C)
            # loop over batch size
            for batch_idx, (src_idxs, target_idxs) in enumerate(indices):
                # loop over objects in one image
                assert len(target_idxs) == max(target_idxs)+1
                ce_mask[batch_idx, :, len(target_idxs):] = 0
                for (src_idx, target_idx) in zip(src_idxs, target_idxs):
                    target_classes_onehot[batch_idx, src_idx, target_idx] = 1
            
            ce_mask = ce_mask.bool()
            
            loss_ce = token_sigmoid_binary_focal_loss(src_logits, target_classes_onehot, text_mask=ce_mask)  / num_masks
            losses = {'loss_VL': loss_ce}

            return losses
        

    def loss_binary_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # [2, 300, 30]
        num_classes = 1
        idx = self._get_src_permutation_idx(indices)
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes_o = torch.zeros_like(target_classes_o)

        target_classes = torch.full(src_logits.shape[:2], num_classes, dtype=torch.int64, device=src_logits.device) #[2,300]
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) #[2, 300, 31]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot,  num_masks, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_binary': loss_ce}
        return losses
    # def loss_labels(self, outputs, targets, indices, num_masks):
    #     """Classification loss (NLL)
    #     targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    #     """
    #     assert "pred_logits" in outputs
    #     src_logits = outputs["pred_logits"].float()

    #     idx = self._get_src_permutation_idx(indices)
    #     target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    #     target_classes = torch.full(
    #         src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
    #     )
    #     target_classes[idx] = target_classes_o

    #     loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
    #     losses = {"loss_ce": loss_ce}
    #     return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]



        with torch.no_grad():
            # sample point_coords
            src_masks_01 = src_masks.clone()
            src_masks_01 = F.interpolate(src_masks_01 , size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
            src_masks_01 = src_masks_to_01(src_masks_01, outputs["binary_class"][src_idx], outputs["pred_iou"][src_idx])
            tgt_iou_scores = calculate_iou(src_masks_01,target_masks)
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        src_iou_scores = outputs["pred_iou"][src_idx].squeeze(1)
        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
            "loss_mask_iou":F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean')
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'binary_labels': self.loss_binary_labels,
            'masks': self.loss_masks,
            'labelsVL': self.loss_labelsVL,
            'loss_language':self.loss_language,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if
                                k != "aux_outputs" and k!="binary_class" and k != "mask2former_outputs"}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()


        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_i = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_i, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "binary_class" in outputs:
            loss = 'binary_labels'
            outputs['pred_logits'] = outputs['binary_class']
            l_dict = self.get_loss(loss, outputs, targets, indices, num_masks )
            losses.update(l_dict)

        if "mask2former_outputs" in outputs:
            loss = 'loss_language'
            l_dict = self.get_loss(loss, outputs["mask2former_outputs"], targets, indices, num_masks )
            losses.update(l_dict)
        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
