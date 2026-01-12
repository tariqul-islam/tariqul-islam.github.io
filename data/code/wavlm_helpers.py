import numpy as np

from transformers import AutoFeatureExtractor, AutoModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel


MODEL_NAME = "microsoft/wavlm-base"

class WavLMFrameRegressor(nn.Module):
    def __init__(self, model_name=MODEL_NAME, out_dim=1, freeze_backbone=False):
        super().__init__()
        #self.backbone = WavLMModel.from_pretrained(model_name)
        self.backbone = WavLMModel.from_pretrained(model_name, use_safetensors=True)

        hidden = self.backbone.config.hidden_size  # 768 for wavlm-base
        self.head = nn.Sequential(
            #nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_values, attention_mask=None):
        out = self.backbone(input_values=input_values, attention_mask=attention_mask)
        frames = out.last_hidden_state                  # (B, T, D)
        pred = self.head(frames).squeeze(-1)           # (B, T) if out_dim=1

        return pred
        
        
def WavLMPreProcessor(model_name=MODEL_NAME):
    return AutoFeatureExtractor.from_pretrained(model_name)
