import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import message_passing as mp

class Model(nn.Module):
  def __init__(self, opt):
    super(Model, self).__init__()
    em_dim = opt.em_dim
    att_dim = opt.att_dim
    vocab_size = opt.vocab_size
    self.DEVICE = opt.DEVICE
    self.lr = opt.lr

    self.em_layer = nn.Embedding(vocab_size, em_dim)
    self.mp_layer = mp.AttIJ(opt)
    self.scorer = mp.Scorers(opt)


    self.params = self.parameters()
    self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=opt.weight_decay)

  def get_output(self, inds):
    em = self.em_layer(inds)
    em_update = self.mp_layer(em)
    scores = self.scorer(em_update)
    return scores

  def get_loss(self, pos, neg):

    pos_sc = self.get_output(pos)
    neg_sc = self.get_output(neg)

    loss = F.softplus(neg_sc-pos_sc)
    loss = loss.mean()
    # print('loss shape: {}'.format(loss.size()))
    self.loss = loss
    return loss


  def train_step(self, pos, neg):
    pos = pos.to(self.DEVICE)
    neg = neg.to(self.DEVICE)
    loss = self.get_loss(pos, neg)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
