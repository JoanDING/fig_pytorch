import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import mp_base as mp
import pdb

class Model(nn.Module):
  def __init__(self, opt):
    super(Model, self).__init__()
    vocab_size = opt.vocab_size
    self.DEVICE = opt.DEVICE
    self.lr = opt.lr
    self.em_layer = nn.Embedding(vocab_size, opt.em_dim)
    if opt.type == 10:
        self.mp_layer = mp.BaseMean_no_mp(opt)
        self.scorer = mp.Scorers_w_id(opt)
        self.params_em = self.em_layer.parameters()
        self.params_sco = self.scorer.parameters()
        self.params = list(self.params_em) + list(self.params_sco)
    elif opt.type == 101602:
        self.mp_layer = mp.BaseMean_10160_2(opt)
        self.scorer = mp.Scorers_w_id(opt)
        self.params_em = self.em_layer.parameters()
        self.params_sco = self.scorer.parameters()
        self.params_mp = self.mp_layer.parameters()
        self.params = list(self.params_em) + list(self.params_mp) + list(self.params_sco)
    elif opt.type == 10160:
        self.mp_layer = mp.BaseMean_10160(opt)
        self.scorer = mp.Scorers_w_id(opt)
        self.params_em = self.em_layer.parameters()
        self.params_sco = self.scorer.parameters()
        self.params_mp = self.mp_layer.parameters()
        self.params = list(self.params_em) + list(self.params_mp) + list(self.params_sco)
    elif opt.type == 101:
        self.mp_layer = mp.BaseMean_add_mp(opt)
        self.scorer = mp.Scorers_w_id(opt)
        self.params_em = self.em_layer.parameters()
        self.params_sco = self.scorer.parameters()
        self.params_mp = self.mp_layer.parameters()
        self.params = list(self.params_em) + list(self.params_mp) + list(self.params_sco)
        #self.params = list(self.params_em)  + list(self.params_sco)
    elif opt.type == 104:
        self.mp_layer = mp.BaseMean_add_mp1(opt)
        self.scorer = mp.Scorers_w_id(opt)
        self.params_em = self.em_layer.parameters()
        self.params_sco = self.scorer.parameters()
        self.params_mp = self.mp_layer.parameters()
        self.params = list(self.params_em) + list(self.params_mp) + list(self.params_sco)
    elif opt.type == 1:
        self.mp_layer = mp.BaseMean_w_id_1(opt)
    elif opt.type == 12:
        self.mp_layer = mp.BaseMean_w_id_1_2(opt)
    elif opt.type == 13:
        self.mp_layer = mp.BaseMean_w_id_1_3(opt)
    elif opt.type == 14:
        self.mp_layer = mp.BaseMean_w_id_1_4(opt)
        opt.em_dim = 2 * opt.em_dim
    elif opt.type == 15:
        self.mp_layer = mp.BaseMean_w_id_1_5(opt)
        opt.em_dim = 2 * opt.em_dim
    elif opt.type == 16:
        self.mp_layer = mp.BaseMean_w_id_1_6(opt)
    elif opt.type == 2:
        self.mp_layer = mp.BaseMean_w_id_2(opt)
    elif opt.type == 3:
        self.mp_layer = mp.BaseMean_w_id_3(opt)
    elif opt.type == 4:
        self.mp_layer = mp.BaseMean_w_id_4(opt)
    elif opt.type == 5:
        self.mp_layer = mp.BaseMean_w_id_5(opt)
        opt.em_dim = 2 * opt.em_dim
    elif opt.type == 6:
        self.mp_layer = mp.BaseMean_w_id_6(opt)
        self.scorer = mp.Scorers_w_id(opt)
        self.params_em = self.em_layer.parameters()
        self.params_sco = self.scorer.parameters()
        self.params = list(self.params_em) + list(self.params_sco)

    self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=opt.weight_decay)
    #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size = 100, gamma = 0.5, last_epoch = -1)

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
    #self.scheduler.step(epoch)
    self.optimizer.step()

