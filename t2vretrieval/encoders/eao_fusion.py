import torch
import torch.nn as nn
import framework.configbase
from timm.models.layers import trunc_normal_
import t2vretrieval
from t2vretrieval.encoders.layers import get_projection
from t2vretrieval.encoders.fusion_transformer import FusionTransformer

class EAOFusionConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.projection_dim = 256
    self.use_positional_embed = True
    self.individual_projections = False
    self.projection = "gated"
    self.dim_embed = 512
    self.projection_dim = 256
    self.max_frames_in_video = 20
    self.max_words_in_sent = 30
    self.fusion_params = {}

class EAOFusion(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    if self.config.use_positional_embed:
        self.vid_pos_embed = nn.Parameter(torch.zeros(1, self.config.max_frames_in_video, self.config.dim_embed)) #.cuda()
        self.txt_pos_embed = nn.Parameter(torch.zeros(1, self.config.max_words_in_sent, self.config.dim_embed)) #.cuda()
        trunc_normal_(self.vid_pos_embed, std=.02)
        trunc_normal_(self.txt_pos_embed, std=.02)
    else:
        self.pos_embed = None

    if not self.config.individual_projections:
        self.proj = get_projection(self.config.dim_embed, self.config.projection_dim, self.config.projection) #.cuda()
    else:
        self.video_proj = get_projection(self.config.dim_embed, self.config.projection_dim, self.config.projection) #.cuda()
        self.text_proj = get_projection(self.config.dim_embed, self.config.projection_dim, self.config.projection) #.cuda()

    self.fusion = FusionTransformer(**self.config.fusion_params) #.cuda()

  def _check_and_fix_if_input_empty(self, x, attention_mask):
    nonempty_input_mask = attention_mask.sum(-1) != 0

    # if all tokens of modality is empty, add one masking token
    empty_input_mask = nonempty_input_mask == 0
    n_masking_tokens = 1
    x[empty_input_mask, :n_masking_tokens] = self.fusion.masking_token.type(x.dtype)
    attention_mask[empty_input_mask, :n_masking_tokens] = 1
    return {'all_tokens': x, 'attention_mask': attention_mask, 'nonempty_input_mask': nonempty_input_mask}
    
  def forward(self, video=None, text=None):
    if video is not None:
        video = self._check_and_fix_if_input_empty(video['all_tokens'], video['attention_mask'])
        if self.config.use_positional_embed:
            video['all_tokens'] = video['all_tokens'] + self.vid_pos_embed
        video_f = self.fusion(video=video)['video']
        return video_f
    
    if text is not None:
        text = self._check_and_fix_if_input_empty(text['all_tokens'], text['attention_mask'])
        if self.config.use_positional_embed:
            text['all_tokens'] = text['all_tokens'] + self.txt_pos_embed[:, :text['all_tokens'].shape[1], :]
        text_f = self.fusion(text=text)['text']
        return text_f
    
    
  def forward_fusion_projection(self, video, text):
    if self.config.individual_projections:
        text_proj, video_proj = self.text_proj, self.video_proj
    else:
        text_proj, video_proj = self.proj, self.proj

    output = {}
    output["cap_embeds"] = text_proj(text['embed'])
    output["vid_embeds"] = video_proj(video['embed'])
    return output