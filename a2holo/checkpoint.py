"""Checkpoints hold the model config, the raw weights and their EMA; results use the EMA.

The distributed checkpoints carry the EMA weights only, stored in fp16 (buffers in fp32);
load_model casts them to fp32."""

import os

import torch

from a2holo.model import A2Holo


def save_checkpoint(path, model, ema_weights, **meta):
    """Write to a temporary file first, so a job killed mid-write leaves the previous file intact."""
    ckpt = dict(config=model.config, weights=model.state_dict(), ema_weights=ema_weights, **meta)
    torch.save(ckpt, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_model(path, device):
    """Build the checkpoint's model with its EMA weights; returns (model, ckpt)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = A2Holo(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["ema_weights"])
    return model.eval(), ckpt
