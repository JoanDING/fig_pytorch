import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import mp_base_2 as mp
import pdb

class Model(nn.Module):
  def __init__(self, opt):
    super(Model, self).__init__()
    em_dim = opt.em_dim
    att_dim = opt.att_dim
    vocab_size = opt.vocab_size
    self.DEVICE = opt.DEVICE
    self.lr = opt.lr
    self.em_layer = nn.Embedding(vocab_size, em_dim)
    if opt.type == 6:
        self.mp_layer = mp.base_mp_6(opt)
    elif opt.type == 101:
        self.mp_layer = mp.BaseMean_add_mp_1(opt)
    elif opt.type == 102:
        self.mp_layer = mp.BaseMean_add_mp_2(opt)
    elif opt.type == 103:
        self.mp_layer = mp.BaseMean_add_mp_3(opt)
    elif opt.type == 104:
        self.mp_layer = mp.BaseMean_add_mp_4(opt)
    self.scorer = mp.Scorers_w_id(opt)
    self.params_em = self.em_layer.parameters()
    self.params_sco = self.scorer.parameters()
    self.params_mp = self.mp_layer.parameters()
    self.params = list(self.params_em) + list(self.params_mp) + list(self.params_sco)
    self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=opt.weight_decay)

  def get_output(self, inds, mask):
    inds = inds.to(self.DEVICE)
    mask = mask.to(self.DEVICE)
    em = self.em_layer(inds)
    em_update = self.mp_layer(em)
    #scores = self.scorer_id(em_update[:,0], mask[:,1])
    #scores = self.scorer_a(em_update[:,1:], mask[:,2:])
    scores = self.scorer(em_update, mask[:,1:])
    return scores

  def get_loss(self, pos, neg, mask_pos, mask_neg):
    self.pos_sc = self.get_output(pos, mask_pos)
    self.neg_sc = self.get_output(neg, mask_neg)
    loss = F.softplus(self.neg_sc-self.pos_sc)
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
    #self.optimizer_em.zero_grad()
    loss.backward()
    self.optimizer.step()
    #self.optimizer_em.step()

