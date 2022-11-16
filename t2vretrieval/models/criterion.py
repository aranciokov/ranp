import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.configbase
import framework.ops

def cosine_sim(im, s):
  '''cosine similarity between all the image and sentence pairs
  '''
  inner_prod = im.mm(s.t())
  im_norm = torch.sqrt((im**2).sum(1).view(-1, 1) + 1e-18)
  s_norm = torch.sqrt((s**2).sum(1).view(1, -1) + 1e-18)
  sim = inner_prod / (im_norm * s_norm)
  return sim

class ContrastiveLoss(nn.Module):
  '''compute contrastive loss
  '''
  def __init__(self, margin=0, max_violation=False, direction='bi', topk=1, semi_hard_neg=False):
    '''Args:
      direction: i2t for negative sentence, t2i for negative image, bi for both
    '''
    super(ContrastiveLoss, self).__init__()
    self.margin = margin
    self.max_violation = max_violation
    self.direction = direction
    self.topk = topk
    self.semi_hard_negative = semi_hard_neg
    print(f"Using SEMI-Hard negative = {self.semi_hard_negative}")

  def forward(self, scores, margin=None, average_batch=True, batch_relevance=None, threshold_pos=1.):
    '''
    Args:
      scores: image-sentence score matrix, (batch, batch)
        the same row of im and s are positive pairs, different rows are negative pairs

      batch_relevance: image-sentence relevancy matrix (batch, batch)
    '''

    if margin is None:
      margin = self.margin

    batch_size = scores.size(0)
    diagonal = scores.diag().view(batch_size, 1) # positive pairs

    # mask to clear diagonals which are positive pairs
    pos_masks = torch.eye(batch_size).bool().to(scores.device)

    batch_topk = min(batch_size, self.topk)
    if self.direction == 'i2t' or self.direction == 'bi':
      d1 = diagonal.expand_as(scores) # same collumn for im2s (negative sentence)
      # compare every diagonal score to scores in its collumn
      # caption retrieval
      cost_s = (margin + scores - d1).clamp(min=0)
      cost_s = cost_s.masked_fill(pos_masks, 0)

      if batch_relevance is not None:
        cost_s[batch_relevance >= threshold_pos] = 0

      ##### ---- SEMI-HARD: also cover the negatives which have s(v, q-) > s(v, q)
      # we look for "the most similar negative example which is less similar than the corresponding positive example" (https://arxiv.org/pdf/2007.12749.pdf, page 4)
      if self.semi_hard_negative:
        cost_s[scores > d1] = 0

      if self.max_violation:
        cost_s, _ = torch.topk(cost_s, batch_topk, dim=1)
        cost_s = cost_s / batch_topk
        if average_batch:
          cost_s = cost_s / batch_size
      else:
        if average_batch:
          cost_s = cost_s / (batch_size * (batch_size - 1))
      cost_s = torch.sum(cost_s)

    if self.direction == 't2i' or self.direction == 'bi':
      d2 = diagonal.t().expand_as(scores) # same row for s2im (negative image)
      # compare every diagonal score to scores in its row
      cost_im = (margin + scores - d2).clamp(min=0)
      cost_im = cost_im.masked_fill(pos_masks, 0)

      if batch_relevance is not None:
        cost_im[batch_relevance >= threshold_pos] = 0

      ##### ---- SEMI-HARD: also cover the negatives which have s(v, q-) > s(v, q)
      # we look for "the most similar negative example which is less similar than the corresponding positive example" (https://arxiv.org/pdf/2007.12749.pdf, page 4)
      if self.semi_hard_negative:
        cost_im[scores > d2] = 0

      if self.max_violation:
        cost_im, _ = torch.topk(cost_im, batch_topk, dim=0)
        cost_im = cost_im / batch_topk
        if average_batch:
          cost_im = cost_im / batch_size
      else:
        if average_batch:
          cost_im = cost_im / (batch_size * (batch_size - 1))
      cost_im = torch.sum(cost_im)

    if self.direction == 'i2t':
      return cost_s
    elif self.direction == 't2i':
      return cost_im
    else:
      return cost_s + cost_im

class ContrastiveLossHP(nn.Module):
  '''compute contrastive loss
  '''
  def __init__(self, margin=0, margin_pos=0, max_violation=False, direction='bi', topk=1, semi_hard_neg=False):
    '''Args:
      direction: i2t for negative sentence, t2i for negative image, bi for both
    '''
    super(ContrastiveLossHP, self).__init__()
    self.margin = margin
    self.margin_pos = margin_pos
    self.max_violation = max_violation
    self.direction = direction
    self.topk = topk
    self.semi_hard_negative = semi_hard_neg
    print(f"Using SEMI-Hard negative = {self.semi_hard_negative}")

  def forward(self, scores, margin=None, average_batch=True, batch_relevance=None, threshold_pos=1., margin_pos=None):
    '''
    Args:
      scores: image-sentence score matrix, (batch, batch)
        the same row of im and s are positive pairs, different rows are negative pairs

      batch_relevance: image-sentence relevancy matrix (batch, batch)
    '''

    if margin is None:
      margin = self.margin
    if margin_pos is None:
      margin_pos = self.margin_pos

    batch_size = scores.size(0)
    diagonal = scores.diag().view(batch_size, 1) # positive pairs

    # mask to clear diagonals which are positive pairs
    pos_masks = torch.eye(batch_size).bool().to(scores.device)

    batch_topk = min(batch_size, self.topk)
    if self.direction == 'i2t' or self.direction == 'bi':
      d1 = diagonal.expand_as(scores) # same collumn for im2s (negative sentence)
      # compare every diagonal score to scores in its collumn
      # caption retrieval
      cost_s = (margin + scores - d1).clamp(min=0)
      cost_s = cost_s.masked_fill(pos_masks, 0)

      if batch_relevance is not None:
        cost_s[batch_relevance >= threshold_pos] = 0

      ##### ---- SEMI-HARD: also cover the negatives which have s(v, q-) > s(v, q)
      # we look for "the most similar negative example which is less similar than the corresponding positive example" (https://arxiv.org/pdf/2007.12749.pdf, page 4)
      if self.semi_hard_negative:
        cost_s[scores > d1] = 0

      if self.max_violation:
        cost_s, _ = torch.topk(cost_s, batch_topk, dim=1)
        cost_s = cost_s / batch_topk
        if average_batch:
          cost_s = cost_s / batch_size
      else:
        if average_batch:
          cost_s = cost_s / (batch_size * (batch_size - 1))
      cost_s = torch.sum(cost_s)

      # we want a copy of scores in order to compute the negative-masked version as well
      hp_scores = scores.clone()  # scores are the (v_i, q_j)
      hp_scores[batch_relevance < threshold_pos] = 1  # mask the negatives; positives have now score <= 1
      hp_scores = hp_scores.masked_fill(pos_masks, 1)
      # we want to pick the argmin (hardest positive)
      hp_scores, _ = torch.topk(hp_scores, batch_topk, dim=1, largest=False)
      # -> s(q, v+)
      hp_cost_s = (margin_pos + scores - hp_scores).clamp(min=0)
      # we need to mask hp_cost again because 'scores' is not pos-masked
      hp_cost_s = hp_cost_s.masked_fill(pos_masks, 0)
      if batch_relevance is not None:
        hp_cost_s[batch_relevance >= threshold_pos] = 0
      
      if self.max_violation:
        hp_cost_s, _ = torch.topk(hp_cost_s, batch_topk, dim=1)
        hp_cost_s = hp_cost_s / batch_topk
        if average_batch:
          hp_cost_s = hp_cost_s / batch_size
      else:
        if average_batch:
          hp_cost_s = hp_cost_s / (batch_size * (batch_size - 1))
      hp_cost_s = torch.sum(hp_cost_s)

    if self.direction == 't2i' or self.direction == 'bi':
      d2 = diagonal.t().expand_as(scores) # same row for s2im (negative image)
      # compare every diagonal score to scores in its row
      cost_im = (margin + scores - d2).clamp(min=0)
      cost_im = cost_im.masked_fill(pos_masks, 0)

      if batch_relevance is not None:
        cost_im[batch_relevance >= threshold_pos] = 0

      ##### ---- SEMI-HARD: also cover the negatives which have s(v, q-) > s(v, q)
      # we look for "the most similar negative example which is less similar than the corresponding positive example" (https://arxiv.org/pdf/2007.12749.pdf, page 4)
      if self.semi_hard_negative:
        cost_im[scores > d2] = 0

      if self.max_violation:
        cost_im, _ = torch.topk(cost_im, batch_topk, dim=0)
        cost_im = cost_im / batch_topk
        if average_batch:
          cost_im = cost_im / batch_size
      else:
        if average_batch:
          cost_im = cost_im / (batch_size * (batch_size - 1))
      cost_im = torch.sum(cost_im)

      # we want a copy of scores in order to compute the negative-masked version as well
      hp_scores_im = scores.clone()  # scores are the (v_i, q_j)
      hp_scores_im[batch_relevance < threshold_pos] = 1  # mask the negatives; positives have now score <= 1
      hp_scores_im = hp_scores_im.masked_fill(pos_masks, 1)
      # we want to pick the argmin (hardest positive)
      hp_scores_im, _ = torch.topk(hp_scores_im, batch_topk, dim=0, largest=False)
      # -> s(q, v+)
      hp_cost_im = (margin_pos + scores - hp_scores_im).clamp(min=0)
      # we need to mask hp_cost again because 'scores' is not pos-masked
      hp_cost_im = hp_cost_im.masked_fill(pos_masks, 0)
      if batch_relevance is not None:
        hp_cost_im[batch_relevance >= threshold_pos] = 0
      """hp_cost_im, _ = torch.topk(hp_cost_im, batch_topk, dim=0)
      hp_cost_im = hp_cost_im / batch_topk
      if average_batch:
        hp_cost_im = hp_cost_im / batch_size"""
      
      if self.max_violation:
        hp_cost_im, _ = torch.topk(hp_cost_im, batch_topk, dim=0)
        hp_cost_im = hp_cost_im / batch_topk
        if average_batch:
          hp_cost_im = hp_cost_im / batch_size
      else:
        if average_batch:
          hp_cost_im = hp_cost_im / (batch_size * (batch_size - 1))
    
      hp_cost_im = torch.sum(hp_cost_im)

    if self.direction == 'i2t':
      return cost_s, hp_cost_s
    elif self.direction == 't2i':
      return cost_im, hp_cost_im
    else:
      return cost_s + cost_im, hp_cost_s + hp_cost_im


class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j



class NormSoftmaxRANPLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x, batch_relevance, threshold):
        batch_topk = 1  # looking for hardest positives
        rows, cols = x.shape
        tmp = x.clone()
        tmp[batch_relevance < threshold] = 1.1  # mask the negatives; positives have now score <= 1
        tmp = tmp.masked_fill(torch.eye(rows, dtype=torch.bool, device=x.device), 1.1)
        # we want to pick the argmin (hardest positive)
        _, hp_indices_im = torch.topk(tmp, batch_topk, dim=0, largest=False)
        _, hp_indices_q = torch.topk(tmp, batch_topk, dim=1, largest=False)
        
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)
        
        # sum over "relevant" (R>=thr)
        idiag = torch.cat((i_logsm.diag(), i_logsm[torch.arange(rows, device=x.device), hp_indices_im.squeeze(0)]))  #[batch_relevance >= threshold]
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.cat((j_logsm.diag(), j_logsm[hp_indices_q.squeeze(1), torch.arange(cols, device=x.device)].view(-1)))  #[batch_relevance >= threshold]
        loss_j = jdiag.sum() / len(jdiag)
        
        return - loss_i - loss_j

