import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import message_passing as mp
import pdb

class Model(nn.Module):
  # ---------------------------------------------------------------------------------------
  # default attention setting:
  # attributes embeddings are updated in FIG, item id embedding doesn't
  # concatate item id embedding and attributes embeddings after FIG
  # input the concatate embeddings into attention Module
  # multiply attributes embedding with attention variables
  # predict scores for all embeddings, including item id
  # ---------------------------------------------------------------------------------------

  def __init__(self, opt):
    super(Model, self).__init__()
    em_dim = opt.em_dim
    att_dim = opt.att_dim
    att_type = opt.att_type
    vocab_size = opt.vocab_size
    self.DEVICE = opt.DEVICE
    self.lr = opt.lr
    self.lr_att = opt.lr_att
    self.em_layer = nn.Embedding(vocab_size, em_dim)
    if att_type == 'plus':
        self.att_layer = mp.plus_attention(opt)
    elif att_type == 'mul':
        self.att_layer = mp.mul_attention(opt)
    elif att_type == 'gat_single':
        self.att_layer = mp.GAT_attention_single(opt)
    elif att_type == 'gat_multi':
        self.att_layer = mp.GAT_attention_multi(opt)
    self.scorer = mp.Scorers_w_id(opt)
    self.params_em = self.em_layer.parameters()
    self.params_sco = self.scorer.parameters()
    #self.params_mp = self.mp_layer.parameters()
    self.params_att = self.att_layer.parameters()
    self.params = list(self.params_sco) + list(self.params_em)
    self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=opt.weight_decay)
    self.optimizer_att = torch.optim.Adam(self.params_att, lr=self.lr_att, weight_decay=opt.weight_decay_att)

  def get_output(self, inds, mask):
    inds = inds.to(self.DEVICE)
    mask = mask.to(self.DEVICE)
    em = self.em_layer(inds) # input: uid,iid,att,bs*8 || output: uid_emb,iid_emb,att_emb bs*8*64
    iid_embed = torch.unsqueeze(em[:,1,:], dim=-2)
    #em_update = self.mp_layer(em) # input: uid,iid,att,bs*8*64 || output: att_emb_updated bs*6*64
    #em_update = torch.cat((iid_embed,em_update),dim=-2) # bs*7*64
    em_update = self.att_layer(em) # input: bs*7*64 || output: bs*7*64
    scores = self.scorer(em_update, mask[:,1:]) # input: bs*7*64 || output:bs*1
    return scores

  def get_loss(self, pos, neg, mask_pos, mask_neg):
    pos_sc = self.get_output(pos, mask_pos)
    neg_sc = self.get_output(neg, mask_neg)
    loss = F.softplus(neg_sc-pos_sc)
    loss = loss.mean()
    # print('loss shape: {}'.format(loss.size()))
    self.loss = loss
    return loss


  def train_step(self, pos, neg, mask_pos, mask_neg):
    pos = pos.to(self.DEVICE)
    neg = neg.to(self.DEVICE)
    mask_pos = mask_pos.to(self.DEVICE)
    mask_neg = mask_neg.to(self.DEVICE)
    loss = self.get_loss(pos, neg, mask_pos, mask_neg)
    self.optimizer.zero_grad()
    self.optimizer_att.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.optimizer_att.step()

