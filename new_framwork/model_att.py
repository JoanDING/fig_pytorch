import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import attentive_message_passing as mp
import pdb

class Model(nn.Module):
  # ---------------------------------------------------------------------------------------
  # default attention setting:
  # attributes embeddings are updated in FIG, item id embedding is not
  # concatate item id embedding and attributes embeddings after FIG
  # multiply attributes embedding with attention variables
  # predict scores for all embeddings, including item id
  # ---------------------------------------------------------------------------------------

  def __init__(self, opt):
    super(Model, self).__init__()
    em_dim = opt.em_dim
    att_dim = opt.att_dim
    att_type = opt.type
    vocab_size = opt.vocab_size
    self.DEVICE = opt.DEVICE
    self.lr = opt.lr
    self.lr_att = opt.lr_att
    self.em_layer = nn.Embedding(vocab_size, em_dim)
    if att_type == 10:
        self.att_layer = mp.att0(opt)
    elif att_type == 101:
        self.att_layer = mp.att0_1(opt)
    elif att_type == 1010:
        self.att_layer = mp.att0_1_multiply(opt)
    elif att_type == 1011:
        self.att_layer = mp.att0_11(opt)
    elif att_type == 1012:
        self.att_layer = mp.att0_12(opt)
    elif att_type == 1013:
        self.att_layer = mp.att0_13(opt)
    elif att_type == 1014:
        self.att_layer = mp.att0_14(opt)
    elif att_type == 1015:
        self.att_layer = mp.att0_15(opt)
    elif att_type == 102:
        self.att_layer = mp.att0_2(opt)
    elif att_type == 103:
        self.att_layer = mp.att0_3(opt)
    elif att_type == 104:
        self.att_layer = mp.att0_4(opt)
    elif att_type == 105:
        self.att_layer = mp.att0_5(opt)
    elif att_type == 106:
        self.att_layer = mp.att0_6(opt)
    elif att_type == 107:
        self.att_layer = mp.att0_7(opt)
    elif att_type == 1:
        self.att_layer = mp.att1(opt)
    elif att_type == 11:
        self.att_layer = mp.att1_1(opt)
    elif att_type == 2:
        self.att_layer = mp.att2(opt)
    elif att_type == 3:
        self.att_layer = mp.att3(opt)
    elif att_type == 31:
        self.att_layer = mp.att3_1(opt)
    elif att_type == 32:
        self.att_layer = mp.att3_2(opt)
    elif att_type == 33:
        self.att_layer = mp.att3_3(opt)
    elif att_type == 34:
        self.att_layer = mp.att3_4(opt)
    elif att_type == 35:
        self.att_layer = mp.att3_5(opt)
    elif att_type == 4:
        self.att_layer = mp.att4(opt)
    elif att_type == 41:
        self.att_layer = mp.att4_1(opt)
    elif att_type == 42:
        self.att_layer = mp.att4_2(opt)
    elif att_type == 43:
        self.att_layer = mp.att4_3(opt)
    elif att_type == 5:
        self.att_layer = mp.att5(opt)
    elif att_type == 6:
        self.att_layer = mp.att6(opt)
    elif att_type == 7:
        self.att_layer = mp.att7(opt)
    elif att_type == 8:
        self.att_layer = mp.att8(opt)
    elif att_type == 9:
        self.att_layer = mp.att9(opt)
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

