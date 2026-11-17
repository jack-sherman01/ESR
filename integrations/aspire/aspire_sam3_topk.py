"""Prune image-grounding candidates before costly mask upsampling.

The upstream HTTP service returns the stable score-sorted top five. This
single-image-only optimization preserves those candidates, including ties.
"""
import torch


def retain_public_topk(outputs, k=5):
    probs = (outputs['pred_logits'].sigmoid()
             * outputs['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
    if probs.ndim != 2 or probs.shape[0] != 1:
        raise ValueError('Only the one-image HTTP service is supported')
    order = torch.argsort(probs[0], descending=True, stable=True)[:k]
    result = dict(outputs)
    for field in ('pred_boxes', 'pred_logits', 'pred_masks'):
        result[field] = outputs[field][:, order]
    return result
