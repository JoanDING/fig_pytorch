import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import message_passing as mp
from attention_module import *
import pdb

class Model(nn.Module):
  def __init__(self, opt):
    super(Model, self).__init__()
    em_dim = opt.em_dim
    att_dim = opt.att_dim
    vocab_size = opt.vocab_size
    self.DEVICE = opt.DEVICE
    self.lr = opt.lr
    self.lr_em = opt.lr_em
    self.em_layer = nn.Embedding(vocab_size, em_dim)
    if opt.attention_model == 1:
        self.att_layer = ATT1(opt)
    elif opt.attention_model == 2:
        self.att_layer = ATT2(opt)
    elif opt.attention_model == 3:
        self.att_layer = ATT3(opt)
    self.scorer = mp.Scorers(opt)
    self.params_em = self.em_layer.parameters()
    self.params_sco = self.scorer.parameters()
    self.params_att = self.att_layer.parameters()
    self.optimizer_sco = torch.optim.Adam(self.params_sco, lr=self.lr, weight_decay=opt.weight_decay2)
    self.optimizer_att = torch.optim.Adam(self.params_att, lr=self.lr, weight_decay=opt.weight_decay_att)
    self.optimizer_em = torch.optim.Adam(self.params_em, lr=self.lr_em, weight_decay=opt.weight_decay)

  def get_output(self, inds, mask):
    inds = inds.to(self.DEVICE)
    mask = mask.to(self.DEVICE)
    em = self.em_layer(inds)
    em_update = self.att_layer(em)
    em_update = self.att_layer(em_update)
    scores = self.scorer(em_update[:,1:,:], mask[:,1:])
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
    self.optimizer_sco.zero_grad()
    self.optimizer_em.zero_grad()
    self.optimizer_att.zero_grad()
    loss.backward()
    self.optimizer_sco.step()
    self.optimizer_em.step()
    self.optimizer_att.step()

