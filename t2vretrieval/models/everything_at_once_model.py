from t2vretrieval.models.globalmatch import VISENC, TXTENC, GlobalMatchModel
#from timm.models.layers import trunc_normal_
from torch import nn as nn
import torch
#from t2vretrieval.encoders.layers import get_projection
#from t2vretrieval.encoders.fusion_transformer import FusionTransformer
import numpy as np

import t2vretrieval.encoders.video
import t2vretrieval.encoders.sentence
import t2vretrieval.encoders.eao_fusion

FUSENC = "fusion"

class EverythingAtOnceModelConfig(t2vretrieval.models.globalmatch.GlobalMatchModelConfig):
  def __init__(self):
    super().__init__()
    """self.projection_dim = 256
    self.use_positional_embed = True
    self.individual_projections = False
    self.projection = "gated"
    self.dim_embed = 512
    self.projection_dim = 256
    self.fusion_params = {}"""

    self.subcfgs[VISENC] = t2vretrieval.encoders.video.EAOConfig()
    self.subcfgs[TXTENC] = t2vretrieval.encoders.sentence.EAOConfig()
    self.subcfgs[FUSENC] = t2vretrieval.encoders.eao_fusion.EAOFusionConfig()

class EverythingAtOnceModel(GlobalMatchModel):
    def __init__(self, config, _logger):
        super().__init__(config, _logger)
        """if self.config.use_positional_embed:
            self.vid_pos_embed = nn.Parameter(torch.zeros(1, self.config.max_frames_in_video, self.config.dim_embed)).cuda()
            self.txt_pos_embed = nn.Parameter(torch.zeros(1, self.config.max_words_in_sent, self.config.dim_embed)).cuda()
            trunc_normal_(self.vid_pos_embed, std=.02)
            trunc_normal_(self.txt_pos_embed, std=.02)
        else:
            self.pos_embed = None

        if not self.config.individual_projections:
            self.proj = get_projection(self.config.dim_embed, self.config.projection_dim, self.config.projection).cuda()
        else:
            self.video_proj = get_projection(self.config.dim_embed, self.config.projection_dim, self.config.projection).cuda()
            self.text_proj = get_projection(self.config.dim_embed, self.config.projection_dim, self.config.projection).cuda()

        self.fusion = FusionTransformer(**self.config.fusion_params).cuda()"""
        #print("**--- ^^ TODO missing information about FusionTransformer (would need to create a submodule for it) ^^ ---**")
    
    def print_comp_graph(self, _loss):
        import torchviz
        print("Computing and rendering the computational graph")
        render_params = dict(self.submods[VISENC].named_parameters(),
                             **dict(self.submods[TXTENC].named_parameters()),
                             **dict(self.submods[FUSENC].named_parameters()))
        dot = torchviz.make_dot(_loss, params=render_params)
        dot.render()
        exit(0)
    
    def build_loss(self):
        criterion = t2vretrieval.models.criterion.ContrastiveLoss(
          margin=self.config.margin,  
          max_violation=self.config.max_violation,
          topk=self.config.hard_topk,
          direction=self.config.loss_direction,
          semi_hard_neg=hasattr(self.config, "semi_hard_neg") and self.config.semi_hard_neg)
        return criterion
        
    def build_submods(self):
        submods = {
            VISENC: t2vretrieval.encoders.video.EAOEncoder(self.config.subcfgs[VISENC]),
            TXTENC: t2vretrieval.encoders.sentence.EAOEncoder(self.config.subcfgs[TXTENC]),
            FUSENC: t2vretrieval.encoders.eao_fusion.EAOFusion(self.config.subcfgs[FUSENC])
        }
        return submods

    def forward_video_embed(self, batch_data):
        vid_fts = torch.FloatTensor(batch_data['attn_fts']).to(self.device)
        vid_lens = torch.LongTensor(batch_data['attn_lens']).to(self.device)
        # (batch, max_vis_len, dim_embed)
        vid_embeds, vid_mask = self.submods[VISENC](vid_fts, vid_lens)
        return {
          'all_tokens': vid_embeds,
          'vid_lens': vid_lens,
          'attention_mask': vid_mask
        }

    def forward_text_embed(self, batch_data):
        txt_fts = torch.LongTensor(batch_data['sent_ids']).to(self.device)
        txt_lens = torch.LongTensor(batch_data['sent_lens']).to(self.device)
        # (batch, max_vis_len, dim_embed)
        txt_embeds, txt_mask = self.submods[TXTENC](txt_fts, txt_lens)
        return {
          'all_tokens': txt_embeds,
          'txt_lens': txt_lens,
          'attention_mask': txt_mask
        }
    
    def forward_fusion(self, video=None, text=None):
        if video is not None:
            return self.submods[FUSENC](video=video)
        if text is not None:
            return self.submods[FUSENC](text=text)
        assert False
    
    def forward_fusion_projection(self, video, text):
        return self.submods[FUSENC].forward_fusion_projection(video=video, text=text)
        
    def evaluate_scores(self, tst_reader):
        vid_names, all_scores = [], []
        cap_names = tst_reader.dataset.captions
        for vid_data in tst_reader:
            vid_names.extend(vid_data['names'])
            vid_enc_outs = self.forward_video_embed(vid_data)
            
            """if self.config.use_positional_embed:
                vid_enc_outs['all_tokens'] = vid_enc_outs['all_tokens'] + self.vid_pos_embed
            video = self.fusion(video=vid_enc_outs)['video']"""
            video = self.forward_fusion(video=vid_enc_outs)
            
            all_scores.append([])
            for cap_data in tst_reader.dataset.iterate_over_captions(self.config.tst_batch_size):
                cap_enc_outs = self.forward_text_embed(cap_data)
                
                """if self.config.use_positional_embed:
                    cap_enc_outs['all_tokens'] = cap_enc_outs['all_tokens'] + self.txt_pos_embed[:, :cap_enc_outs['all_tokens'].shape[1], :]
                text = self.fusion(text=cap_enc_outs)['text']"""
                text = self.forward_fusion(text=cap_enc_outs)
                
                """if self.config.individual_projections:
                    text_proj, video_proj = self.text_proj, self.video_proj
                else:
                    text_proj, video_proj = self.proj, self.proj

                output = {}
                output["cap_embeds"] = text_proj(text['embed'])
                output["vid_embeds"] = video_proj(video['embed'])"""
                output = self.forward_fusion_projection(video=video, text=text)

                scores = self.generate_scores(**output)
                # scores = self.generate_phrase_scores(output['vid_embeds'], video['attention_mask'], output['cap_embeds'], text['attention_mask'])
                all_scores[-1].append(scores.data.cpu().numpy())
            all_scores[-1] = np.concatenate(all_scores[-1], axis=1)
        all_scores = np.concatenate(all_scores, axis=0) # (n_img, n_cap)
        return vid_names, cap_names, all_scores

    def generate_scores(self, **kwargs):
        def normalize_embeddings(a, eps=1e-8):
            a_n = a.norm(dim=1)[:, None]
            a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
            return a_norm


        def sim_matrix(a, b, eps=1e-8):
            """
            added eps for numerical stability
            """
            a = normalize_embeddings(a, eps)
            b = normalize_embeddings(b, eps)

            sim_mt = torch.mm(a, b.transpose(0, 1))
            return sim_mt
        
        # compute image-sentence similarity
        vid_embeds = kwargs['vid_embeds']
        cap_embeds = kwargs['cap_embeds']
        scores = sim_matrix(vid_embeds, cap_embeds) # s[i, j] i: im_idx, j: s_idx
        return scores

    def evaluate(self, tst_reader, return_outs=False):
        vid_names, cap_names, scores = self.evaluate_scores(tst_reader)

        if "1kA" in tst_reader.dataset.dname:
            i2t_gts = np.arange(len(vid_names)).reshape(len(vid_names), 1).tolist()
        else:
            i2t_gts = []
            for vid_name in vid_names:
              i2t_gts.append([])
              for i, cap_name in enumerate(cap_names):
                if cap_name in tst_reader.dataset.ref_captions[vid_name]:
                  i2t_gts[-1].append(i)

        t2i_gts = {}
        for i, t_gts in enumerate(i2t_gts):
          for t_gt in t_gts:
            t2i_gts.setdefault(t_gt, [])
            t2i_gts[t_gt].append(i)

        metrics = self.calculate_metrics(scores, i2t_gts, t2i_gts)
        
        if not "1kA" in tst_reader.dataset.dname:
            rel_mat = tst_reader.dataset.relevance_matrix
            print(f"using relevance matrix with shape {rel_mat.shape}")

            from .ndcg_map_helpers import calculate_k_counts, calculate_mAP, calculate_nDCG, calculate_IDCG
            vis_k_counts = calculate_k_counts(rel_mat)
            txt_k_counts = calculate_k_counts(rel_mat.T)

            idcg_v = calculate_IDCG(rel_mat, vis_k_counts)
            idcg_t = calculate_IDCG(rel_mat.T, txt_k_counts)

            vis_nDCG = calculate_nDCG(scores, rel_mat, vis_k_counts, IDCG=idcg_v)
            txt_nDCG = calculate_nDCG(scores.T, rel_mat.T, txt_k_counts, IDCG=idcg_t)
            print('nDCG: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_nDCG, txt_nDCG, (vis_nDCG + txt_nDCG) / 2))

            vis_mAP = calculate_mAP(scores, rel_mat)
            txt_mAP = calculate_mAP(scores.T, rel_mat.T)
            print('mAP: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_mAP, txt_mAP, (vis_mAP + txt_mAP) / 2))

            metrics["nDCG_vt"] = vis_nDCG
            metrics["nDCG_tv"] = txt_nDCG
            metrics["nDCG_avg"] = (vis_nDCG + txt_nDCG) / 2
            metrics["mAP_vt"] = vis_mAP
            metrics["mAP_tv"] = txt_mAP
            metrics["mAP_avg"] = (vis_mAP + txt_mAP) / 2

        if return_outs:
          outs = {
            'vid_names': vid_names,
            'cap_names': cap_names,
            'scores': scores,
          }
          return metrics, outs
        else:
          return metrics

    def forward_loss(self, batch_data, step=None):
        vid_enc_outs = self.forward_video_embed(batch_data)
        cap_enc_outs = self.forward_text_embed(batch_data)
        # add positional embedding after masking
        """if self.config.use_positional_embed:
            vid_enc_outs['all_tokens'] = vid_enc_outs['all_tokens'] + self.vid_pos_embed
            cap_enc_outs['all_tokens'] = cap_enc_outs['all_tokens'] + self.txt_pos_embed[:, :cap_enc_outs['all_tokens'].shape[1], :]

        text = self.fusion(text=cap_enc_outs)['text']
        video = self.fusion(video=vid_enc_outs)['video']

        if self.config.individual_projections:
            text_proj, video_proj = self.text_proj, self.video_proj
        else:
            text_proj, video_proj = self.proj, self.proj
        
        output = {}
        output["cap_embeds"] = text_proj(text['embed'])
        output["vid_embeds"] = video_proj(video['embed'])"""

        video = self.forward_fusion(video=vid_enc_outs)
        text = self.forward_fusion(text=cap_enc_outs)
        output = self.forward_fusion_projection(video=video, text=text)

        scores = self.generate_scores(**output)
        # scores = self.generate_phrase_scores(output['vid_embeds'], video['attention_mask'], output['cap_embeds'], text['attention_mask'])
        loss = self.criterion(scores)
        
        # self.print_comp_graph(loss)

        if step is not None and self.config.monitor_iter > 0 and step % self.config.monitor_iter == 0:
          neg_scores = scores.masked_fill(torch.eye(len(scores), dtype=torch.bool).to(self.device), -1e10)
          self.print_fn('\tstep %d: pos mean scores %.2f, hard neg mean scores i2t %.2f, t2i %.2f'%(
            step, torch.mean(torch.diag(scores)), torch.mean(torch.max(neg_scores, 1)[0]), 
            torch.mean(torch.max(neg_scores, 0)[0])))
        
        self.logger.add_scalar("train/loss", loss.item(), step)
        return loss


class EverythingAtOnceNormSoftModel(EverythingAtOnceModel):
    def build_loss(self):
        criterion = t2vretrieval.models.criterion.NormSoftmaxLoss(self.config.temperature)
        return criterion


class EverythingAtOnceWord2VecModel(EverythingAtOnceModel):
    def build_submods(self):
        submods = {
            VISENC: t2vretrieval.encoders.video.EAOEncoder(self.config.subcfgs[VISENC]),
            TXTENC: t2vretrieval.encoders.sentence.EAOWord2VecEncoder(self.config.subcfgs[TXTENC]),
            FUSENC: t2vretrieval.encoders.eao_fusion.EAOFusion(self.config.subcfgs[FUSENC])
        }
        return submods

    def forward_text_embed(self, batch_data):
        txt_fts = torch.FloatTensor(batch_data['sent_ids']).to(self.device)
        txt_lens = torch.FloatTensor(batch_data['sent_lens']).to(self.device)
        # (batch, max_vis_len, dim_embed)
        txt_embeds, txt_mask = self.submods[TXTENC](txt_fts, txt_lens)
        return {
          'all_tokens': txt_embeds,
          'txt_lens': txt_lens,
          'attention_mask': txt_mask
        }


class EverythingAtOnceNormSoftW2VModel(EverythingAtOnceWord2VecModel):
    def build_loss(self):
        criterion = t2vretrieval.models.criterion.NormSoftmaxLoss(self.config.temperature)
        return criterion


class EverythingAtOnceRANPModel(EverythingAtOnceModel):
    def build_loss(self):
        criterion = t2vretrieval.models.criterion.ContrastiveLossHP(
          margin=self.config.margin,
          margin_pos=self.config.margin_pos,
          max_violation=self.config.max_violation,
          topk=self.config.hard_topk,
          direction=self.config.loss_direction,
          semi_hard_neg=hasattr(self.config, "semi_hard_neg") and self.config.semi_hard_neg)
        return criterion
  
    def forward_loss(self, batch_data, step=None):
        vid_enc_outs = self.forward_video_embed(batch_data)
        cap_enc_outs = self.forward_text_embed(batch_data)

        video = self.forward_fusion(video=vid_enc_outs)
        text = self.forward_fusion(text=cap_enc_outs)
        output = self.forward_fusion_projection(video=video, text=text)

        scores = self.generate_scores(**output)
        # scores = self.generate_phrase_scores(output['vid_embeds'], video['attention_mask'], output['cap_embeds'], text['attention_mask'])
        
        from .ndcg_map_helpers import get_relevances_single_caption, get_relevances_multi_caption
        threshold_pos = batch_data["threshold_pos"]
        assert threshold_pos < 1
        noun_classes = batch_data['noun_classes']
        verb_class = batch_data['verb_class']
        # get_relevances should be dataset-dependent
        #  (for MSR->multi-captions-per-clip->
        #  it should not be "batch captions"x"batch captions", but "batch-videos' 'pooled-captions'"x"batch caption")
        #  --> video_verb_classes, video_noun_classes from batch_data
        if "video_verb_classes" in batch_data:
            video_verb_classes = batch_data["video_verb_classes"]
            video_noun_classes = batch_data["video_noun_classes"]

            sent_rel = get_relevances_multi_caption(video_verbs=video_verb_classes, video_nouns=video_noun_classes,
                                                    batch_verbs=verb_class, batch_nouns=noun_classes).cuda()

        else:
            sent_rel = get_relevances_single_caption(batch_verbs=verb_class, batch_nouns=noun_classes).cuda()

        base_loss, hp_loss = self.criterion(scores, batch_relevance=sent_rel, threshold_pos=threshold_pos)
        
        loss = base_loss + hp_loss
        
        if step is not None and self.config.monitor_iter > 0 and step % self.config.monitor_iter == 0:
          neg_scores = scores.masked_fill(torch.eye(len(scores), dtype=torch.bool).to(self.device), -1e10)
          self.print_fn('\tstep %d: pos mean scores %.2f, hard neg mean scores i2t %.2f, t2i %.2f'%(
            step, torch.mean(torch.diag(scores)), torch.mean(torch.max(neg_scores, 1)[0]), 
            torch.mean(torch.max(neg_scores, 0)[0])))
        
        self.logger.add_scalar("train/loss", loss.item(), step)
        return loss

class EverythingAtOnceRANPWord2VecModel(EverythingAtOnceRANPModel):
    def build_submods(self):
        submods = {
            VISENC: t2vretrieval.encoders.video.EAOEncoder(self.config.subcfgs[VISENC]),
            TXTENC: t2vretrieval.encoders.sentence.EAOWord2VecEncoder(self.config.subcfgs[TXTENC]),
            FUSENC: t2vretrieval.encoders.eao_fusion.EAOFusion(self.config.subcfgs[FUSENC])
        }
        return submods

    def forward_text_embed(self, batch_data):
        txt_fts = torch.FloatTensor(batch_data['sent_ids']).to(self.device)
        txt_lens = torch.FloatTensor(batch_data['sent_lens']).to(self.device)
        # (batch, max_vis_len, dim_embed)
        txt_embeds, txt_mask = self.submods[TXTENC](txt_fts, txt_lens)
        return {
          'all_tokens': txt_embeds,
          'txt_lens': txt_lens,
          'attention_mask': txt_mask
        }


class EverythingAtOnceRANModel(EverythingAtOnceModel):
    def build_loss(self):
        criterion = t2vretrieval.models.criterion.ContrastiveLoss(
          margin=self.config.margin,
          max_violation=self.config.max_violation,
          topk=self.config.hard_topk,
          direction=self.config.loss_direction,
          semi_hard_neg=hasattr(self.config, "semi_hard_neg") and self.config.semi_hard_neg)
        return criterion
  
    def forward_loss(self, batch_data, step=None):
        vid_enc_outs = self.forward_video_embed(batch_data)
        cap_enc_outs = self.forward_text_embed(batch_data)

        video = self.forward_fusion(video=vid_enc_outs)
        text = self.forward_fusion(text=cap_enc_outs)
        output = self.forward_fusion_projection(video=video, text=text)

        scores = self.generate_scores(**output)
        # scores = self.generate_phrase_scores(output['vid_embeds'], video['attention_mask'], output['cap_embeds'], text['attention_mask'])
        
        from .ndcg_map_helpers import get_relevances_single_caption, get_relevances_multi_caption
        threshold_pos = batch_data["threshold_pos"]
        assert threshold_pos < 1
        noun_classes = batch_data['noun_classes']
        verb_class = batch_data['verb_class']
        # get_relevances should be dataset-dependent
        #  (for MSR->multi-captions-per-clip->
        #  it should not be "batch captions"x"batch captions", but "batch-videos' 'pooled-captions'"x"batch caption")
        #  --> video_verb_classes, video_noun_classes from batch_data
        if "video_verb_classes" in batch_data:
            video_verb_classes = batch_data["video_verb_classes"]
            video_noun_classes = batch_data["video_noun_classes"]

            sent_rel = get_relevances_multi_caption(video_verbs=video_verb_classes, video_nouns=video_noun_classes,
                                                    batch_verbs=verb_class, batch_nouns=noun_classes).cuda()

        else:
            sent_rel = get_relevances_single_caption(batch_verbs=verb_class, batch_nouns=noun_classes).cuda()
        
        
        loss = self.criterion(scores, batch_relevance=sent_rel, threshold_pos=threshold_pos)
        
        if step is not None and self.config.monitor_iter > 0 and step % self.config.monitor_iter == 0:
          neg_scores = scores.masked_fill(torch.eye(len(scores), dtype=torch.bool).to(self.device), -1e10)
          self.print_fn('\tstep %d: pos mean scores %.2f, hard neg mean scores i2t %.2f, t2i %.2f'%(
            step, torch.mean(torch.diag(scores)), torch.mean(torch.max(neg_scores, 1)[0]), 
            torch.mean(torch.max(neg_scores, 0)[0])))
        
        self.logger.add_scalar("train/loss", loss.item(), step)
        return loss

class EverythingAtOnceRANWord2VecModel(EverythingAtOnceRANModel):
    def build_submods(self):
        submods = {
            VISENC: t2vretrieval.encoders.video.EAOEncoder(self.config.subcfgs[VISENC]),
            TXTENC: t2vretrieval.encoders.sentence.EAOWord2VecEncoder(self.config.subcfgs[TXTENC]),
            FUSENC: t2vretrieval.encoders.eao_fusion.EAOFusion(self.config.subcfgs[FUSENC])
        }
        return submods

    def forward_text_embed(self, batch_data):
        txt_fts = torch.FloatTensor(batch_data['sent_ids']).to(self.device)
        txt_lens = torch.FloatTensor(batch_data['sent_lens']).to(self.device)
        # (batch, max_vis_len, dim_embed)
        txt_embeds, txt_mask = self.submods[TXTENC](txt_fts, txt_lens)
        return {
          'all_tokens': txt_embeds,
          'txt_lens': txt_lens,
          'attention_mask': txt_mask
        }


class EverythingAtOnceNormSoftRANPModel(EverythingAtOnceWord2VecModel):
    def build_loss(self):
        criterion = t2vretrieval.models.criterion.NormSoftmaxRANPLoss(self.config.temperature)
        return criterion
  
    def forward_loss(self, batch_data, step=None):
        vid_enc_outs = self.forward_video_embed(batch_data)
        cap_enc_outs = self.forward_text_embed(batch_data)

        video = self.forward_fusion(video=vid_enc_outs)
        text = self.forward_fusion(text=cap_enc_outs)
        output = self.forward_fusion_projection(video=video, text=text)

        scores = self.generate_scores(**output)
        # scores = self.generate_phrase_scores(output['vid_embeds'], video['attention_mask'], output['cap_embeds'], text['attention_mask'])
        
        from .ndcg_map_helpers import get_relevances_single_caption, get_relevances_multi_caption
        threshold_pos = batch_data["threshold_pos"]
        assert threshold_pos < 1
        noun_classes = batch_data['noun_classes']
        verb_class = batch_data['verb_class']
        # get_relevances should be dataset-dependent
        #  (for MSR->multi-captions-per-clip->
        #  it should not be "batch captions"x"batch captions", but "batch-videos' 'pooled-captions'"x"batch caption")
        #  --> video_verb_classes, video_noun_classes from batch_data
        if "video_verb_classes" in batch_data:
            video_verb_classes = batch_data["video_verb_classes"]
            video_noun_classes = batch_data["video_noun_classes"]

            sent_rel = get_relevances_multi_caption(video_verbs=video_verb_classes, video_nouns=video_noun_classes,
                                                    batch_verbs=verb_class, batch_nouns=noun_classes).cuda()

        else:
            sent_rel = get_relevances_single_caption(batch_verbs=verb_class, batch_nouns=noun_classes).cuda()

        loss = self.criterion(scores, sent_rel, threshold_pos)
        
        if step is not None and self.config.monitor_iter > 0 and step % self.config.monitor_iter == 0:
          neg_scores = scores.masked_fill(torch.eye(len(scores), dtype=torch.bool).to(self.device), -1e10)
          self.print_fn('\tstep %d: pos mean scores %.2f, hard neg mean scores i2t %.2f, t2i %.2f'%(
            step, torch.mean(torch.diag(scores)), torch.mean(torch.max(neg_scores, 1)[0]), 
            torch.mean(torch.max(neg_scores, 0)[0])))
        
        self.logger.add_scalar("train/loss", loss.item(), step)
        return loss
