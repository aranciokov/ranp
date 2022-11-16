import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.configbase

class MPEncoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.dim_fts = [2048]
    self.dim_embed = 1024
    self.dropout = 0

class MPEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    input_size = sum(self.config.dim_fts)
    self.ft_embed = nn.Linear(input_size, self.config.dim_embed, bias=True)
    self.dropout = nn.Dropout(self.config.dropout)

  def forward(self, inputs):
    '''
    Args:
      inputs: (batch, dim_fts) or (batch, max_seq_len, dim_fts)
    Return:
      embeds: (batch, dim_embed) or (batch, max_seq_len, dim_fts)
    '''
    embeds = self.ft_embed(inputs)
    embeds = self.dropout(embeds)
    return embeds


class EAOConfig(framework.configbase.ModuleConfig):
    def __init__(self):
        super().__init__()
        self.dim_fts = [2048]
        self.dim_embed = 512
        self.use_positional_embed = True
        self.token_projection = "gated"

class EAOEncoder(nn.Module):
    def __init__(self, config):
        from t2vretrieval.encoders.layers import get_projection

        super().__init__()
        self.config = config
        self.norm_layer = nn.LayerNorm(self.config.dim_embed, eps=1e-6)
        
        self.token_proj = get_projection(sum(self.config.dim_fts), self.config.dim_embed, self.config.token_projection)
            
    def forward(self, x, lens=None):
        y = self.token_proj(x)
        y = self.norm_layer(y)
        if lens is not None:
            assert len(lens.shape) == 1, f"Lengths should be 1-D, found {lens.shape}"
            mask = (torch.arange(y.shape[1], device=y.device).repeat(y.shape[0], 1) < lens.unsqueeze(1)).float()
            y = y * mask.clone().unsqueeze(-1)
        
        return y, mask